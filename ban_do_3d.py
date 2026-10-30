"""
ban_do_3d.py — Trực quan hoá 3D mạng SAGSIN mở rộng từ dữ liệu v5.

    python ban_do_3d.py                             # bão soulik, 20 khung phủ cả cơn
    python ban_do_3d.py --bao bualoi                # đổi cơn
    python ban_do_3d.py --bao manh-nhat             # cơn mưa nhiều nhất
    python ban_do_3d.py --thoi-tiet-binh-thuong     # thời tiết cả năm
    python ban_do_3d.py --in dataset/v5/val.npz --bao trami --frames 30
    python ban_do_3d.py --liet-ke                   # liệt kê tệp dữ liệu + 10 cơn bão

Cấu trúc dữ liệu xuất ra (đọc bởi assets_ban_do/ve_3d.js):
  meta   names, kinds, src, dst, ground, island, series
  frame  t, p ([lat, lon, alt_km] * N), isl, acc, adapt, R, G, ac, link,
         cap ([tổng cặp, cặp tới được]), co_lap, so_tp,
         brk (cạnh đứt), brk_mua (đứt do mưa), brk_nut (đứt do nút bị tắt), down

Nguồn dữ liệu:
  dataset/v5/{test,val}.npz    W_snapshots, node_*, y_paths_*, link_stats, reach_stats
  data/bao_manh/*.npz          10 cơn bão: precip, gust, cape tại 74 điểm
  data/bao_manh/danh_muc.json  danh mục 10 cơn (tên, giờ UTC, mốc kịch bản)
  data/thoi_tiet_vn_mr.npz     thời tiết cả năm
  data/bao_manh/bang_tra/meta.json  tham số mức "khắc nghiệt"

Đứt gãy tính bằng src/ngau_nhien/kich_ban.py (sinh_kich_ban): từ quan_sat['song']
và quan_sat['tat'] tách được cạnh đứt do mưa (SNR < snr_min_db) và cạnh đứt do
một đầu bị tắt (UAV hạ cánh, trạm/HAPS gặp sét).
"""


from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

for _luong in (sys.stdout, sys.stderr):          # console Windows hay mặc định cp1258
    try:
        _luong.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

GOC = Path(__file__).resolve().parent
ASSET = GOC / "assets_ban_do"
BAO_DIR = GOC / "data" / "bao_manh"
BAO_DANH_MUC = BAO_DIR / "danh_muc.json"
THOI_TIET = GOC / "data" / "thoi_tiet_vn_mr.npz"

# Thứ tự loại nút — PHẢI khớp KSTYLE trong ve_3d.js
KIND = {"leo": 0, "haps": 1, "uav": 2, "gateway": 3, "user": 4}
SAT, AIR, GROUND = "space", "air", "ground"

# Tên các tệp bão có trong data/bao_manh — dùng để chạy lại bộ mô phỏng lấy đứt gãy.
BANG_XAC_NHAN = {
    # 10 cơn bão mạnh (data/bao_manh/*.npz)
    "prapiroon_2024", "soulik_2024", "trami_2024", "wutip_2025", "wipha_2025",
    "kajiki_2025", "nongfa_2025", "bualoi_2025", "matmo_2025", "kalmaegi_2025",
    # thời tiết cả năm (data/thoi_tiet_vn_mr.npz) — cũng chạy được cùng bộ mô phỏng
    "thoi_tiet_vn_mr",
}


# Quần đảo — toạ độ trùng src/physics/sites.py (EXTRA_ISLAND_SITES)
# Nút thuộc quần đảo (toạ độ lấy từ node_latlon_alt của dataset, trùng sites.py).
DAO = ("HoangSa", "TruongSa")

TEN_VN = {
    "Hanoi": "Hà Nội", "DaNang": "Đà Nẵng", "HoChiMinh": "TP.HCM", "HaiPhong": "Hải Phòng",
    "CanTho": "Cần Thơ", "Vinh": "Vinh", "Hue": "Huế", "NhaTrang": "Nha Trang",
    "QuyNhon": "Quy Nhơn", "BuonMaThuot": "Buôn Ma Thuột", "CaMau": "Cà Mau",
    "LaoCai": "Lào Cai", "PleiKu": "Pleiku", "KonTum": "Kon Tum", "DaLat": "Đà Lạt",
    "SaPa": "Sa Pa", "NamDinh": "Nam Định", "ThaiBinh": "Thái Bình", "ThanhHoa": "Thanh Hóa",
    "QuangBinh": "Quảng Bình", "QuangTri": "Quảng Trị", "QuangNinh": "Quảng Ninh",
    "VanDon": "Vân Đồn", "HaiHa": "Hải Hà", "HaiHau": "Hải Hậu", "GiaoThuy": "Giao Thủy",
    "BachLongVi": "Bạch Long Vĩ", "ConDao": "Côn Đảo", "LySon": "Lý Sơn",
    "PhuQuoc": "Phú Quốc", "HoangSa": "Hoàng Sa", "TruongSa": "Trường Sa",
    "HAPS-North": "HAPS Bắc", "HAPS-Central": "HAPS Trung", "HAPS-South": "HAPS Nam",
    "HAPS-Highland": "HAPS Tây Nguyên", "HAPS-Mekong": "HAPS Mê Kông",
    "HAPS-BacTrungBo": "HAPS Bắc Trung Bộ", "HAPS-BienDong": "HAPS Biển Đông",
}


# ---------------------------------------------------------------------------
# Rà soát dữ liệu (không đọc phần nặng nếu không cần)
# ---------------------------------------------------------------------------

UNG_VIEN = [
    ("v5/test  — 224 nút, 150 Starlink phủ VN (bản mới nhất)", "dataset/v5/test.npz"),
    ("v5/val   — 224 nút", "dataset/v5/val.npz"),
    ("v5/train — 224 nút, 294 snapshot (nặng hơn)", "dataset/v5/train.npz"),
    ("v4/test  — 330 nút, chòm Walker mô phỏng", "dataset/v4/test.npz"),
    ("v3/test  — 174 nút", "dataset/v3/test.npz"),
    ("demo n373 — 373 nút (không có node_latlon_alt)", "huong_dan/du_lieu_demo/n373.npz"),
    ("demo n273 — 273 nút (không có node_latlon_alt)", "huong_dan/du_lieu_demo/n273.npz"),
    ("demo n173 — 173 nút (không có node_latlon_alt)", "huong_dan/du_lieu_demo/n173.npz"),
    ("demo n73  — 73 nút (không có node_latlon_alt)", "huong_dan/du_lieu_demo/n73.npz"),
]

# Tệp còn trong dự án nhưng KHÔNG vẽ được (giải thích trong --liet-ke)
KHONG_VE = [
    ("dataset/test.npz", "bộ v2 73 nút: không có node_latlon_alt"),
    ("dataset/train.npz", "bộ v2: không có node_latlon_alt"),
    ("dataset/val.npz", "bộ v2: không có node_latlon_alt"),
    ("dataset/oracle_dataset_v2.npz", "bản gốc v2: không có node_latlon_alt"),
    ("dataset/oracle_dataset_v3.npz", "bản gốc v3 — dùng dataset/v3/{test,val}.npz"),
    ("dataset/oracle_dataset_v4.npz", "bản gốc v4 — dùng dataset/v4/{test,val}.npz"),
]


def doc_khoa(path: Path, *khoa: str) -> dict:
    """Chỉ nạp các khoá cần (tránh link_stats của oracle gốc nặng hàng trăm MB)."""
    out = {}
    with np.load(path, allow_pickle=False) as raw:
        for k in khoa:
            if k in raw.files:
                out[k] = raw[k]
    return out


def kiem_tra(path: Path) -> tuple[bool, str]:
    """Tệp có vẽ được không + lý do."""
    if not path.exists():
        return False, "không có tệp"
    try:
        d = doc_khoa(path, "W_snapshots", "node_order", "node_latlon_alt",
                     "snapshot_id", "y_paths_indices")
    except Exception as e:
        return False, f"không đọc được ({e})"
    if "W_snapshots" not in d:
        return False, "thiếu W_snapshots (chỉ là đặc trưng/mẫu, không phải mạng)"
    W = d["W_snapshots"]
    if W.ndim != 3 or W.shape[1] != W.shape[2]:
        return False, f"W_snapshots shape {W.shape} không phải [S, N, N]"
    n = W.shape[1]
    if "node_order" not in d or len(d["node_order"]) != n:
        return False, "node_order không khớp N"
    if "node_latlon_alt" not in d:
        return False, "thiếu node_latlon_alt (không dựng được toạ độ)"
    la = np.asarray(d["node_latlon_alt"])
    if la.ndim != 3 or la.shape[1:] != (n, 3):
        return False, f"node_latlon_alt shape {la.shape} không khớp N={n}"
    for k in ("snapshot_id", "y_paths_indices"):
        if k not in d:
            return False, f"thiếu {k}"
    if int((W > 0).sum()) == 0:
        return False, "W_snapshots không có liên kết nào"
    return True, f"S={W.shape[0]}, N={n}, {len(d['y_paths_indices'])} mẫu"


def in_liet_ke() -> None:
    print(f"{'TỆP':<44}{'TRẠNG THÁI':<13}GHI CHÚ")
    for ten, rel in UNG_VIEN:
        ok, ly_do = kiem_tra(GOC / rel)
        print(f"{rel:<44}{('DÙNG ĐƯỢC' if ok else 'BỎ QUA'):<13}{ly_do}")
    for rel, vi_sao in KHONG_VE:
        if (GOC / rel).exists():
            print(f"{rel:<44}{'BỎ QUA':<13}{vi_sao}")
    for p in (THOI_TIET,):
        if p.exists():
            d = doc_khoa(p, "site_names", "time_utc", "precip", "gust", "cape")
            co = [k for k in ("precip", "gust", "cape") if k in d]
            print(f"{str(p.relative_to(GOC)):<44}{'THỜI TIẾT':<13}"
                  f"{len(d['site_names'])} điểm, {len(d['time_utc'])} giờ (cột: {', '.join(co)})")
    bao = doc_bao()
    if bao:
        print(f"\n{'BÃO (data/bao_manh)':<44}{'KHẮC NGHIỆT':<13}giờ  mưa max  gió max")
        for b in bao:
            print(f"  {b['ten']:<42}{len(b['time']):>4}{float(b['precip'].max()):>9.1f}"
                  f"{float(b['gust'].max()):>9.1f}")
    else:
        print("\nKhông đọc được data/bao_manh/danh_muc.json")
    print("\nToạ độ: node_latlon_alt có sẵn trong v5/v3/v4 nên KHÔNG cần dựng lại từ TLE.")


def tim_nguon(ten: str | None) -> Path:
    if ten:
        p = Path(ten)
        if not p.is_absolute():
            p = GOC / p
        ok, ly_do = kiem_tra(p)
        if not ok:
            raise SystemExit(f"'{ten}' không vẽ được: {ly_do}\nChạy --liet-ke để xem tệp dùng được.")
        return p
    for _ten, rel in UNG_VIEN:
        p = GOC / rel
        if kiem_tra(p)[0]:
            return p
    raise SystemExit("Không có tệp nào vẽ được. Chạy --liet-ke để xem chi tiết.")


# ---------------------------------------------------------------------------
# Thời tiết
# ---------------------------------------------------------------------------


def doc_bao() -> list[dict]:
    """Danh mục 10 cơn bão mạnh (data/bao_manh) — mỗi cơn là một tệp thời tiết đã cắt.

    Đây mới là dữ liệu thời tiết KHẮC NGHIỆT: mưa tới 43 mm/h, gió giật 38.7 m/s,
    CAPE 5060 — so với thời tiết bình thường trong thoi_tiet_vn_mr.npz (mưa tối đa 15 mm/h).
    """
    if not BAO_DANH_MUC.exists():
        return []
    try:
        cat = json.loads(BAO_DANH_MUC.read_text(encoding="utf-8"))
    except Exception:
        return []
    ra = []
    for b in cat.get("bao", []):
        p = BAO_DIR / str(b.get("tep", ""))
        if not p.exists():
            continue
        d = doc_khoa(p, "site_names", "time_utc", "precip", "gust", "cape")
        if not all(k in d for k in ("site_names", "time_utc", "precip", "gust")):
            continue
        ra.append({
            "ten": str(b.get("ten", p.stem)),
            "ten_tep": p.stem,
            "tep": f"data/bao_manh/{p.name}",
            "names": [str(x) for x in d["site_names"]],
            "time": [str(x) for x in d["time_utc"]],
            "precip": np.nan_to_num(np.asarray(d["precip"], np.float32)),
            "gust": np.nan_to_num(np.asarray(d["gust"], np.float32)),
            "cape": np.nan_to_num(np.asarray(d.get("cape", np.zeros_like(d["precip"])), np.float32)),
            "giong": True,
            "gio_bat_dau": list(b.get("gio_bat_dau_kich_ban", [])) or None,
        })
    return ra


def doc_thoi_tiet() -> dict:
    """Thời tiết bình thường (cả năm) — dùng khi muốn xem lúc trời không bão."""
    if THOI_TIET.exists():
        d = doc_khoa(THOI_TIET, "site_names", "time_utc", "precip", "gust", "cape")
        return {
            "ten": "thời tiết cả năm (bình thường)",
            "tep": str(THOI_TIET.relative_to(GOC)),
            "ten_tep": THOI_TIET.stem,   # để chạy dut_gay() như với bão
            "names": [str(x) for x in d["site_names"]],
            "time": [str(x) for x in d["time_utc"]],
            "precip": np.nan_to_num(np.asarray(d["precip"], np.float32)),
            "gust": np.nan_to_num(np.asarray(d["gust"], np.float32)),
            "cape": np.nan_to_num(np.asarray(d.get("cape", np.zeros_like(d["precip"])), np.float32)),
            "gio_bat_dau": None,
        }
    print("  ! Không thấy tệp thời tiết — mưa/gió để 0")
    z = np.zeros((1, 0), np.float32)
    return {"ten": "không có", "tep": None, "names": [], "time": [], "precip": z, "gust": z,
            "cape": z, "gio_bat_dau": None}


def chon_nguon_thoi_tiet(ten_bao: str | None, uu_tien_bao: bool,
                         n_snap: int = 1, diem_ground: list[str] | None = None) -> dict:
    """--bao <tên> | --bao manh-nhat | (mặc định) cơn nặng nhất
    | --thoi-tiet-binh-thuong."""
    danh_sach = doc_bao()
    for b in danh_sach:
        b["_ground"] = set(diem_ground or [])
    # --thoi-tiet-binh-thuong ưu tiên hơn mặc định của --bao
    if not uu_tien_bao:
        return doc_thoi_tiet()
    if ten_bao is None:
        return doc_thoi_tiet()
    if not danh_sach:
        print("  ! Không đọc được data/bao_manh — dùng thời tiết bình thường")
        return doc_thoi_tiet()
    if ten_bao and ten_bao not in ("manh-nhat", "auto"):
        for b in danh_sach:
            if b["ten"] == ten_bao or b["ten"].startswith(ten_bao):
                return b
        raise SystemExit(f"Không có bão '{ten_bao}'. Có: "
                         f"{', '.join(b['ten'] for b in danh_sach)}, tat-ca")
    # cơn "nặng nhất" = tổng mưa lớn nhất trên các điểm MẶT ĐẤT
    return max(danh_sach, key=lambda b: float(
        b["precip"][:, [k for k, nm in enumerate(b["names"]) if nm in b["_ground"]]].sum()
        if b["_ground"] else b["precip"].sum()))


def gio_mua_nhat(tt: dict, n_frame: int = 1, khoang: int = 1) -> int:
    """Giờ bắt đầu khung hình.

    Bão dài 25–101 giờ, nên chỉ tiến 1 giờ mỗi khung thì cả 10 khung vẫn nằm trong vài
    giờ đầu và không thấy bão lớn lên. Ở chế độ bão, `khoang` được tính lại cho phủ trọn
    cơn: mỗi khung cách nhau `len(time)/n_frame` giờ, bắt đầu từ giờ 0 của cơn. Nhờ vậy
    chuỗi khung đi hết: hình thành → đỉnh → tan.

    Với dữ liệu bình thường: giờ có nhiều điểm mưa nhất trong ngày.
    """
    if tt["tep"] is None or not tt["time"]:
        return -1
    if tt.get("gio_bat_dau") is not None:      # là một cơn bão: phủ trọn cơn
        return 0
    g = np.arange(0, len(tt["time"]), 24)
    dem = (tt["precip"][g] > 2.5).sum(1)
    return int(g[int(np.argmax(dem))])


def thoi_diem(tt: dict, gio: int, sid: int) -> str:
    """Giờ VN (UTC+7) của cột thời tiết đang dùng."""
    if 0 <= gio < len(tt["time"]):
        raw = tt["time"][gio]
        for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
                return (dt + timedelta(hours=7)).strftime("%d/%m %H:%M")
            except ValueError:
                pass
        return raw[5:16]
    return f"snapshot {sid}"


# ---------------------------------------------------------------------------
# Liên kết
# ---------------------------------------------------------------------------


def loai_cap(layer: np.ndarray) -> np.ndarray:
    """(N,N) int8: 0 = ISL (vệ tinh–vệ tinh), 1 = GSL, 2 = LOCAL."""
    space = layer == SAT
    t = np.full((layer.size, layer.size), 2, np.int8)
    t[space[:, None] & space[None, :]] = 0
    t[space[:, None] & ~space[None, :]] = 1
    t[~space[:, None] & space[None, :]] = 1
    return t


def bang_canh(W: np.ndarray, loai: np.ndarray) -> dict:
    iu, iv = np.nonzero(np.triu(W > 0, 1))
    return {"u": iu, "v": iv, "w": W[iu, iv].astype(np.float64), "loai": loai[iu, iv]}


def cap_phang(u: np.ndarray, v: np.ndarray) -> list[int]:
    """[u0, v0, u1, v1, ...] — dạng mảng phẳng mà ve_3d.js đọc."""
    out = np.empty(u.size * 2, np.int32)
    out[0::2] = u
    out[1::2] = v
    return out.tolist()


# ---------------------------------------------------------------------------
# Dựng các khung hình
# ---------------------------------------------------------------------------



def loc_canh(cap: dict, loai_can: int, dut: set) -> list[int]:
    """Cặp [u,v,...] của cạnh thuộc `loai_can`, bỏ các cạnh đã đứt.

    Cạnh đứt vẽ riêng màu đỏ; nếu vẫn nằm trong lớp sống thì bị vẽ chồng hai màu.
    """
    u, v, t = cap["u"], cap["v"], cap["loai"]
    giu = (t == loai_can) & np.array([(int(a), int(b)) not in dut for a, b in zip(u, v)])
    return cap_phang(u[giu], v[giu])


def thanh_phan(W0: np.ndarray) -> tuple[np.ndarray, int]:
    """Nhãn thành phần liên thông của đồ thị liên kết thật (W0 > 0).

    Trả (nhãn từng nút, số thành phần). Nút đứng một mình là thành phần riêng — đó
    chính là "nút không nối được với ai" theo dữ liệu, không phải suy diễn.
    """
    n = W0.shape[0]
    nhan = np.full(n, -1, np.int32)
    k = 0
    for st in range(n):
        if nhan[st] >= 0:
            continue
        nhan[st] = k
        q = [st]
        while q:
            u = q.pop()
            for v in np.where(W0[u] > 0)[0]:
                if nhan[v] < 0:
                    nhan[v] = k
                    q.append(int(v))
        k += 1
    return nhan, k


def chon_snapshot(n_frame: int, snapshot: int | None, n_snap: int) -> list[int]:
    tam = n_snap // 2 if snapshot is None else snapshot
    if not 0 <= tam < n_snap:
        raise SystemExit(f"--snapshot {tam} ngoài [0, {n_snap - 1}]")
    dau = min(max(0, tam - n_frame // 2), max(0, n_snap - n_frame))
    return list(range(dau, min(dau + n_frame, n_snap)))


def tuyen_theo_snapshot(data: dict, n: int) -> dict[int, np.ndarray]:
    """Tuyến oracle đầu tiên của mỗi snapshot (chỉ số nút, đã lọc -1)."""
    y = np.asarray(data["y_paths_indices"])
    sid = np.asarray(data["snapshot_id"])
    out: dict[int, np.ndarray] = {}
    for m in range(y.shape[0]):
        s = int(sid[m])
        if s in out:
            continue
        pr = y[m][(y[m] >= 0) & (y[m] < n)]
        if pr.size >= 2:
            out[s] = pr.astype(int)
    return out


def dung_luoi(data: dict, tt: dict, n_frame: int, snapshot: int | None,
              khoang: int, ten_bao: str | None = None) -> dict:
    """Mỗi khung ứng với một snapshot của dataset; giờ thời tiết đi kèm.

    Chế độ bình thường: các snapshot liên tiếp, giờ tiến `khoang` giờ mỗi khung.
    Chế độ nhiều bão: `<tt>["moc"]` đã là danh sách (snapshot, giờ) dựng sẵn — dùng nguyên.
    """
    names = [str(x) for x in data["node_order"]]
    n = len(names)
    layer = np.asarray(data["node_layer"]).astype(str) if "node_layer" in data else \
        np.where(np.char.startswith(np.char.upper(np.asarray(names)), "STARLINK"), SAT, GROUND)
    kind_s = np.asarray(data["node_kind"]).astype(str) if "node_kind" in data else \
        np.where(layer == SAT, "leo", "user")
    kinds = [KIND.get(k, 4) for k in kind_s]
    loai = loai_cap(layer)
    la = np.asarray(data["node_latlon_alt"], np.float32)
    W = np.asarray(data["W_snapshots"], np.float32)
    reach = np.asarray(data["reach_stats"]) if "reach_stats" in data else None
    h0_bao = 0
    if ten_bao:
        try:
            _cat = json.loads((GOC / 'data' / 'bao_manh' / 'danh_muc.json')
                              .read_text(encoding='utf-8'))
            _b = next((x for x in _cat['bao'] if x['ten'] == ten_bao), None)
            if _b and _b.get('gio_bat_dau_kich_ban'):
                h0_bao = int(_b['gio_bat_dau_kich_ban'][0])
        except Exception:
            h0_bao = 0

    # link_stats: số liệu liên kết thật (khoảng cách, góc nâng, băng thông, mất gói).
    # Mỗi liên kết vô hướng ghi 2 hàng (u->v, v->u) nên tra theo cả hai chiều.
    ls_idx, c_dist, c_elev, c_bw, c_loss = {}, 0, 1, 2, 3
    if "link_stats" in data and "link_stat_cols" in data:
        cols = [str(c) for c in np.asarray(data["link_stat_cols"])]
        def _cot(ten, mac_dinh):
            return cols.index(ten) if ten in cols else mac_dinh
        c_dist = _cot("distance_km", 3)
        c_elev = _cot("elevation_deg", 4)
        c_bw = _cot("bandwidth_mbps", 5)
        c_loss = _cot("packet_loss", 6)
        ls = np.asarray(data["link_stats"])
        if ls.size:
            for row in ls:
                ls_idx[(int(row[0]), int(row[1]), int(row[2]))] = row

    temps = tuyen_theo_snapshot(data, n)

    ground = np.where(layer == GROUND)[0]
    site_of = np.full(ground.size, -1)
    for k, i in enumerate(ground):
        if names[i] in tt["names"]:
            site_of[k] = tt["names"].index(names[i])
    co_tt = np.where(site_of >= 0)[0]
    air_ground = [i for i in range(n) if layer[i] != SAT]        # UAV + HAPS + ground

    # Danh sách (snapshot, giờ thời tiết, nhãn) cho từng khung
    moc = tt.get("moc")
    if moc:
        ke_hoach = [(int(s), int(g), lab) for s, g, lab in moc]
    else:
        frames_orig = chon_snapshot(n_frame, snapshot, W.shape[0])
        bo = gio_mua_nhat(tt, n_frame)
        so_gio = len(tt["time"])
        # Ở chế độ bão: rải khung đều khắp cơn. Không dùng phép % vì sẽ quay vòng
        # về đầu cơn khi số khung × bước vượt quá độ dài cơn.
        if bo == 0 and tt.get("gio_bat_dau") is not None and n_frame > 1:
            buoc = max(1, int(round(so_gio / n_frame)))
        else:
            buoc = khoang
        cuoi = so_gio - 1
        trai_deu = buoc != khoang      # rải đều khắp cơn -> chia đúng khoảng mốc

        def _gio(fi: int) -> int:
            if bo < 0:
                return -1
            if trai_deu:
                return 0 if n_frame <= 1 else round(fi * cuoi / (n_frame - 1))
            return min(bo + fi * buoc, cuoi)

        ke_hoach = [(s, _gio(fi), thoi_diem(tt, _gio(fi), s))
                    for fi, s in enumerate(frames_orig)]

    # Mẫu đầu tiên của khung hình: nguồn/đích để tô nổi bật
    y = np.asarray(data["y_paths_indices"])
    sid = np.asarray(data["snapshot_id"])
    src = dst = 0
    for s, _g, _l in ke_hoach:
        m = np.where(sid == s)[0]
        if m.size:
            pr = y[m[0]][y[m[0]] >= 0]
            if pr.size >= 2:
                src, dst = int(pr[0]), int(pr[-1])
                break

    frames, chuoi_ac, chuoi_nb = [], [], []
    for s, gio, nhan in ke_hoach:
        W0 = W[s]
        cap = bang_canh(W0, loai)

        # Mưa / gió theo MẶT ĐẤT — dữ liệu thật từ tệp bão, khớp theo tên điểm.
        R = np.zeros(ground.size, np.float32)
        G = np.zeros(ground.size, np.float32)
        if gio >= 0 and co_tt.size:
            R[co_tt] = tt["precip"][gio, site_of[co_tt]]
            G[co_tt] = tt["gust"][gio, site_of[co_tt]]

        tuyen = temps.get(s)
        route = tuyen.tolist() if tuyen is not None else []
        adapt = [x for q in range(len(route) - 1) for x in (route[q], route[q + 1])]

        # Chi phí tuyến = tổng trễ của các chặng, lấy thẳng từ W. Không có đủ chặng thì bỏ.
        ac = None
        if len(route) >= 2:
            w_seg = [float(W0[a, b]) for a, b in zip(route[:-1], route[1:])]
            if all(w > 0 for w in w_seg):
                ac = round(sum(w_seg), 2)

        # link_stats của v5 cho từng chặng: khoảng cách, góc nâng, băng thông, mất gói.
        thong_so_tuyen: list[float] = []
        for a, b in zip(route[:-1], route[1:]):
            r = ls_idx.get((s, int(a), int(b)))
            if r is None:
                r = ls_idx.get((s, int(b), int(a)))
            if r is None:
                thong_so_tuyen += [0.0, 0.0, 0.0, 0.0]
            else:
                thong_so_tuyen += [round(float(r[c_dist]), 1), round(float(r[c_elev]), 1),
                                   round(float(r[c_bw]), 0), round(float(r[c_loss]), 4)]
        chuoi_ac.append(ac)
        chuoi_nb.append(int(cap["u"].size))

        # Chạy lại bộ mô phỏng để tách cạnh đứt theo nguyên nhân:
        # mưa (SNR thấp) hay một đầu bị tắt.
        dut_mua, dut_nut, nut_tat = [], [], []
        if ten_bao and ten_bao in BANG_XAC_NHAN:
            try:
                dut_mua, dut_nut, nut_tat, _ = dut_gay(W0, data, ten_bao, s, h0_bao)
            except Exception as e:      # thiếu thư viện / tệp: bỏ qua, không chặn bản đồ
                print(f"  ! không tính được đứt gãy cho snapshot {s}: {e}")

        # --- Dữ liệu THẬT về "không có đường đi" ---
        # reach_stats[s] = [snapshot_id, n_pairs, n_reachable]   (SCHEMA.md dòng 41)
        tong_cap = toi_duoc = 0
        if reach is not None and s < len(reach):
            tong_cap, toi_duoc = int(reach[s, 1]), int(reach[s, 2])
        nhan_tp, so_tp = thanh_phan(W0)
        dem_tp = np.bincount(nhan_tp)
        co_lap = [int(i) for i in range(n) if dem_tp[nhan_tp[i]] == 1]

        # Tập cạnh đứt để loại khỏi lớp sống (tránh vẽ chồng hai màu).
        dut_set = {(a, b) for a, b in zip(dut_mua[0::2] + dut_nut[0::2],
                                          dut_mua[1::2] + dut_nut[1::2])}

        frames.append({
            "t": nhan,
            "p": np.round(la[s].reshape(-1), 4).tolist(),
            "isl": loc_canh(cap, 0, dut_set),
            "acc": loc_canh(cap, 1, dut_set) + loc_canh(cap, 2, dut_set),
            # Đường đi tách theo nút bị tắt: chặng còn dùng được / chặng đứt.
            "adapt": [x for q in range(0, len(adapt), 2)
                      if adapt[q] not in set(nut_tat) and adapt[q + 1] not in set(nut_tat)
                      for x in (adapt[q], adapt[q + 1])],
            "adapt_dut": [x for q in range(0, len(adapt), 2)
                          if adapt[q] in set(nut_tat) or adapt[q + 1] in set(nut_tat)
                          for x in (adapt[q], adapt[q + 1])],
            "R": np.round(R, 2).tolist(),
            "G": np.round(G, 2).tolist(),
            "ac": ac,
            "link": thong_so_tuyen,
            # reach_stats: [tổng cặp, cặp tới được]; hiệu là số cặp mất đường.
            "cap": [tong_cap, toi_duoc],
            "co_lap": co_lap,
            "so_tp": int(so_tp),
            # Đứt gãy thật (W=-1) tách theo nguyên nhân:
            "brk": dut_mua + dut_nut,      # tất cả cạnh đứt -> vẽ đỏ
            "brk_mua": dut_mua,            # đứt vì mưa
            "brk_nut": dut_nut,            # đứt vì nút bị tắt
            "down": nut_tat,               # nút đang bị tắt (UAV/trạm/HAPS)
        })

    island = [1 if nm in DAO else 0 for nm in names]

    return {
        "meta": {
            "names": names,
            "kinds": kinds,
            "src": src,
            "dst": dst,
            "ground": [int(i) for i in ground],
            "island": island,
            "stepSec": int(np.asarray(data.get("step_sec", [20])).ravel()[0])
            if "step_sec" in data else 20,
            "muc": "thời tiết thật",
            "nhan": 1,
            "event": "",
            "series": {"ac": chuoi_ac, "bc": chuoi_ac, "nb": chuoi_nb},
        },
        "frames": frames,
    }



# ---------------------------------------------------------------------------
# Đứt gãy thật: chạy lại bộ mô phỏng của dự án (src/ngau_nhien/kich_ban.py)
# ---------------------------------------------------------------------------


def tham_so_khac_nghiet() -> "ThamSo":
    """Tham số mức 'khắc nghiệt' lấy nguyên từ data/bao_manh/bang_tra/meta.json."""
    from src.ngau_nhien.kich_ban import ThamSo

    meta = json.loads((GOC / "data" / "bao_manh" / "bang_tra" / "meta.json")
                      .read_text(encoding="utf-8"))
    t = meta["parameters"]
    return ThamSo(
        mua=t["mua"], giong=t["giong"], xung_dot=t["xung_dot"], tai=t["tai"],
        dt_s=t["dt_s"], phi_mua=t["phi_mua"], sigma_mua=t["sigma_mua"],
        snr0_db=t["snr0_db"], snr_min_db=t["snr_min_db"],
        gio_giat_max=t["gio_giat_max"], uav_ha_canh=t["uav_ha_canh"],
        p_set=t["p_set"], set_tb=t["set_tb"],
    )


def dut_gay(W0: np.ndarray, data: dict, ten_bao: str, snapshot: int,
            h0: int, buoc: int = 40):
    """Chạy sinh_kich_ban() trên một snapshot, trả về 2 danh sách cặp đứt.

    Dùng lặp snapshot thành `buoc` bước thời gian để mô phỏng dao động mưa/gió;
    lấy bước giữa làm đại diện. Trả (dut_mua, dut_nut, nut_tat) là danh sách chỉ số
    nút dạng [u,v,u,v,...] và danh sách nút bị tắt tại bước đó.
    """
    from src.ngau_nhien.kich_ban import MangGoc, sinh_kich_ban
    from src.ngau_nhien.thoi_tiet import ThoiTiet

    names = [str(x) for x in data["node_order"]]
    kind = [str(x) for x in data["node_kind"]]
    tep = GOC / "data" / "bao_manh" / f"{ten_bao}.npz"
    if not tep.exists():                      # thời tiết cả năm
        tep = GOC / "data" / f"{ten_bao}.npz"
    tt = ThoiTiet(tep)
    W3 = np.repeat(W0[None], buoc, axis=0)          # cùng hình học, thời tiết dao động
    g = MangGoc(W3, kind, names, tt)
    Wn, qs = sinh_kich_ban(g, tt, int(h0), 12345, tham_so_khac_nghiet())

    t = buoc // 2
    tat, song = qs["tat"][t], qs["song"][t]
    co_goc = g.W[t] > 0
    hai_dau_tat = tat[:, None] | tat[None, :]
    mua = co_goc & ~song & ~hai_dau_tat
    nut = co_goc & ~song & hai_dau_tat

    def cap(mask):
        iu, iv = np.nonzero(np.triu(mask, 1))
        out = np.empty(iu.size * 2, np.int32)
        out[0::2], out[1::2] = iu, iv
        return out.tolist(), iv

    dm = cap(mua)[0]
    iu, iv = np.nonzero(np.triu(nut, 1))
    dn = np.empty(iu.size * 2, np.int32)
    dn[0::2], dn[1::2] = iu, iv
    return dm, dn.tolist(), [int(i) for i in np.where(tat)[0]], int(snapshot)


# ---------------------------------------------------------------------------
# Xuất HTML
# ---------------------------------------------------------------------------


def doc_bo_cuc() -> tuple[str, str, str]:
    dau, than, ve = (ASSET / "khung_dau.txt"), (ASSET / "khung_than.txt"), (ASSET / "ve_3d.js")
    if not all(p.exists() for p in (dau, than, ve)):
        raise SystemExit(f"Thiếu mẫu giao diện trong {ASSET} (khung_dau.txt, khung_than.txt, ve_3d.js)")
    d = dau.read_text(encoding="utf-8")
    t = than.read_text(encoding="utf-8")
    v = ve.read_text(encoding="utf-8")
    if 'id="gl"' not in t or "const COUNTRIES" not in t:
        raise SystemExit("khung_than.txt hỏng")
    if "const M = D.meta" not in v:
        raise SystemExit("ve_3d.js hỏng")
    return d, t, v


def xuat_html(D: dict, ra: Path) -> None:
    d, t, v = doc_bo_cuc()
    js = json.dumps(D, ensure_ascii=False, separators=(",", ":"))
    html = d + t + "const D = " + js + ";\n" + v
    ra.parent.mkdir(parents=True, exist_ok=True)
    ra.write_text(html, encoding="utf-8", newline="")
    print(f"Đã tạo: {ra}  ({ra.stat().st_size / 1e6:.1f} MB)")


def main() -> Path:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="vao", default=None, help="tệp .npz (mặc định dataset/v5/test.npz)")
    ap.add_argument("--out", default="ban_do_3d.html", help="tệp HTML xuất ra")
    ap.add_argument("--snapshot", type=int, default=None, help="snapshot giữa khung hình")
    ap.add_argument("--khoang-gio", type=int, default=1, help="mỗi khung tiến bao nhiêu giờ")
    ap.add_argument("--bao", default="soulik",
                    help="cơn bão cần vẽ (yagi, soulik, bualoi, trami, wutip, wipha, kajiki, "
                         "nongfa, matmo, kalmaegi) hoặc 'manh-nhat' = cơn mưa nhiều nhất; "
                         "")
    ap.add_argument("--frames", type=int, default=0,
                    help="số khung; ở chế độ bão rải đều khắp cơn (mặc định 40)")
    ap.add_argument("--thoi-tiet-binh-thuong", action="store_true",
                    help="dùng thời tiết cả năm thay vì bão")
    ap.add_argument("--liet-ke", action="store_true", help="liệt kê tệp dữ liệu và trạng thái")
    ap.add_argument("--khong-mo", action="store_true", help="không tự mở trình duyệt")
    a = ap.parse_args()

    if a.liet_ke:
        in_liet_ke()
        return Path()

    src = tim_nguon(a.vao)
    print(f"Đọc: {src.relative_to(GOC) if src.is_relative_to(GOC) else src}")
    data = doc_khoa(src, "W_snapshots", "node_order", "node_layer", "node_kind",
                    "node_latlon_alt", "snapshot_id", "y_paths_indices",
                    "link_stats", "link_stat_cols", "reach_stats")
    # Mặc định dùng BÃO (dữ liệu khắc nghiệt thật); --thoi-tiet-binh-thuong để xem cả năm.
    n_snap = np.asarray(data["W_snapshots"]).shape[0]
    layer = np.asarray(data["node_layer"]).astype(str)
    gnd = [str(nm) for nm, ly in zip(data["node_order"], layer) if ly == GROUND]
    tt = chon_nguon_thoi_tiet(a.bao, uu_tien_bao=not a.thoi_tiet_binh_thuong,
                              n_snap=n_snap, diem_ground=gnd)
    print(f"Thời tiết: {tt['ten']} ({tt['tep'] or 'không có'})")
    # Số khung: mỗi khung một mốc riêng, giới hạn bởi số snapshot của dataset và
    # số giờ của tệp thời tiết (mỗi giờ là một hàng).
    n_khung = a.frames or min(n_snap, len(tt["time"]))
    print(f"Số khung: {n_khung}")
    D = dung_luoi(data, tt, n_khung, a.snapshot, a.khoang_gio,
                  ten_bao=tt.get('ten_tep') or None)
    meta = D["meta"]
    meta["event"] = (f"{len(meta['names'])} nút · {len(D['frames'])} khung · "
                     f"thời tiết: {tt['ten']}")
    ra = Path(a.out)
    if not ra.is_absolute():
        ra = GOC / ra
    xuat_html(D, ra)
    k = meta["kinds"]
    print("  " + ", ".join(f"{ten} {k.count(v)}" for ten, v in
                           [("vệ tinh", 0), ("HAPS", 1), ("UAV", 2), ("gateway", 3), ("user", 4)]))
    f0 = D["frames"][0]
    rmax = max(f0["R"]) if f0["R"] else 0.0
    print(f"  liên kết: {(len(f0['acc']) + len(f0['isl'])) // 2} | "
          f"mưa lớn nhất {rmax:.1f} mm/h | trễ tuyến {f0['ac']} ms")
    if not a.khong_mo:
        webbrowser.open(ra.resolve().as_uri())
    return ra


if __name__ == "__main__":
    main()
