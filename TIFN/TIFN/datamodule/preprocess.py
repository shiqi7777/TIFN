import iisignature
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from iisignature import siglength, sig, logsig, logsiglength

_PREP_CACHE = {}
def _get_logsig_prep(d, k):
    """缓存 iisignature.prepare 的预编译结果"""
    key = (d, k)
    if key not in _PREP_CACHE:
        _PREP_CACHE[key] = iisignature.prepare(d, k)
    return _PREP_CACHE[key]

def interpolate_stroke(stroke, num_points=128):
    stroke = np.array(stroke)
    if len(stroke) < 2:
        return stroke
    distances = np.cumsum(np.sqrt(np.sum(np.diff(stroke, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0)
    new_distances = np.linspace(0, distances[-1], num_points)
    new_stroke = np.zeros((num_points, 2))
    new_stroke[:, 0] = np.interp(new_distances, distances, stroke[:, 0])
    new_stroke[:, 1] = np.interp(new_distances, distances, stroke[:, 1])
    return new_stroke

def resample_by_arclen_adaptive(xy_norm: np.ndarray, max_points: int,
                                ds_norm: float = 1/128.0, min_points: int = 1) -> np.ndarray:
    xy_norm = np.asarray(xy_norm, dtype=np.float32)
    T = len(xy_norm)
    if T < 1:
        return np.zeros((1, 2), dtype=np.float32)
    if T == 1:
        return xy_norm[:1]

    dif = np.diff(xy_norm, axis=0)
    seg = np.sqrt((dif**2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    if L < 1e-6:
        return xy_norm[:1]

    n = int(np.ceil(L / ds_norm)) + 1
    n = max(min_points, min(max_points, n))

    u = np.linspace(0.0, L, n, dtype=np.float32)
    x = np.interp(u, s, xy_norm[:, 0]).astype(np.float32)
    y = np.interp(u, s, xy_norm[:, 1]).astype(np.float32)
    return np.stack([x, y], axis=-1)

def extract_signature_feature(
    traces,
    k: int = 2,               # k=2 => D = siglength(input_dim, k)
    window_size: int = 4,     # 以当前点为中心的滑窗半径
    interp_points: int = 128, # 每条笔画插值到固定步数
    norm: str = "isotropic",     # or "zscore"
    use_time: bool = False
):
    """
    返回:
      seq_feat: np.ndarray[T, D]
      使用 [x, y, t] 作为签名输入维度（D = siglength(3, k)）
    """
    # 1) 过滤并插值到固定长度
    raw_strokes = [np.asarray(st, dtype=np.float32) for st in traces if len(st) >= 1]
    if not raw_strokes:
        D = logsiglength(3, k)  # 3维输入（x,y,t）
        return np.zeros((1, D), dtype=np.float32)

    # 全局归一化（仅 x,y）
    all_pts = np.vstack(raw_strokes)
    if norm == "isotropic":
        min_xy = all_pts.min(axis=0)
        max_xy = all_pts.max(axis=0)

        center = (min_xy + max_xy) / 2.0
        scale = np.max(max_xy - min_xy)
        scale = max(scale, 1e-6)

        normed = [(s - center) / scale for s in raw_strokes]

    elif norm == "zscore":
        mu = all_pts.mean(axis=0)
        sd = np.maximum(all_pts.std(axis=0), 1e-6)
        normed = [(s - mu) / sd for s in raw_strokes]
    elif norm == "minmax":
        mn, mx = all_pts.min(0), all_pts.max(0)
        rng = np.maximum(mx - mn, 1e-6)
        normed = [(s - mn) / rng for s in raw_strokes]
    else:
        raise ValueError(f"Unknown normalization mode: {norm}")

    # 自适应重采样（上限 interp_points）
    ds_norm = 1 / 128.0
    strokes = [resample_by_arclen_adaptive(s, max_points=interp_points, ds_norm=ds_norm, min_points=1) for s in normed]

    # 构造时间通道 t∈[0,1]
    traj = []
    if use_time:
        # ====== 带时间版本 ======
        for s in strokes:
            if len(s) == 1:
                t = np.zeros((1, 1), dtype=np.float32)
            else:
                dif = np.diff(s, axis=0)
                seg = np.sqrt((dif ** 2).sum(axis=1))
                arc_len = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
                total = float(max(arc_len[-1], 1e-6))
                t = (arc_len / total).reshape(-1, 1)
            traj.append(np.concatenate([s, t], axis=-1))  # [T,3]
        d_in = 3
    else:
        # ====== 不带时间版本 ======
        traj = strokes
        d_in = 2

    # 滑窗签名（输入维=3）
    # D = siglength(d_in, k)
    D = logsiglength(d_in, k)
    prep = _get_logsig_prep(d_in, k)  # 预编译的 spec
    seq_list = []
    for s3 in traj:
        T = len(s3)
        feat = np.zeros((T, D), dtype=np.float32)
        for i in range(T):
            i0 = max(0, i - window_size)
            i1 = min(T, i + window_size + 1)
            if i1 - i0 >= 2:
                sub = s3[i0:i1].astype(np.float64, copy=False)
                sub = np.ascontiguousarray(sub)
                v = logsig(sub, prep)
                # v = sig(sub, k)
                feat[i] = v.astype(np.float32, copy=False)
        seq_list.append(feat)

    return np.concatenate(seq_list, axis=0)  # [sum_T, D]

# def extract_signature_feature(
#     traces,
#     k: int = 2,               # k=2 => D = 6
#     window_size: int = 4,     # 以当前点为中心的滑窗半径
#     interp_points: int = 128, # 每条笔画插值到固定步数
#     norm: str = "minmax",     # or "zscore"
# ):
#     """
#     返回:
#       seq_feat: np.ndarray[T, D]
#     """
#     # 1) 每条笔画插值
#     strokes = [np.asarray(st, dtype=np.float32) for st in traces if len(st) >= 1]
#     D = siglength(2, k)
#     if not strokes:
#         return np.zeros((1, D), dtype=np.float32)
#
#     # 2) 归一化到稳定尺度
#     all_pts = np.vstack(strokes)
#     if norm == "minmax":
#         min_xy = all_pts.min(axis=0)
#         max_xy = all_pts.max(axis=0)
#         rng = np.maximum(max_xy - min_xy, 1e-6)
#         strokes = [(s - min_xy) / rng for s in strokes]
#     else:  # zscore
#         mu = all_pts.mean(axis=0)
#         std = np.maximum(all_pts.std(axis=0), 1e-6)
#         strokes = [(s - mu) / std for s in strokes]
#
#     # 3) 在每条笔画内做滑窗签名（不跨越笔画边界）
#     D = siglength(2, k)
#     seq_list = []
#     for s in strokes:
#         T = len(s)
#         sig_seq = np.zeros((T, D), dtype=np.float32)
#         for i in range(T):
#             i0 = max(0, i - window_size)
#             i1 = min(T, i + window_size + 1)
#             if i1 - i0 >= 2:
#                 sub = s[i0:i1]               # [n, 2]
#                 sig_seq[i] = sig(sub, k)     # [D]
#             # 边缘不足窗口长度会得到 0 向量
#         seq_list.append(sig_seq)
#     return np.concatenate(seq_list, axis=0)  # [sum_T, D]


class Segment(object):
    """Class to reprsent a Segment compound of strokes (id) with an id and label."""

    __slots__ = ("id", "label", "strId")

    def __init__(self, *args):
        if len(args) == 3:
            self.id = args[0]
            self.label = args[1]
            self.strId = args[2]
        else:
            self.id = "none"
            self.label = ""
            self.strId = set([])


class Inkml(object):
    """Class to represent an INKML file with strokes, segmentation and labels"""

    __slots__ = ("fileName", "strokes", "strkOrder", "segments", "truth", "UI")

    NS = {
        "ns": "http://www.w3.org/2003/InkML",
        "xml": "http://www.w3.org/XML/1998/namespace",
    }

    def __init__(self, *args):
        self.fileName = None
        self.strokes = {}
        self.strkOrder = []
        self.segments = {}
        self.truth = ""
        self.UI = ""
        if len(args) == 1:
            self.fileName = args[0]
            self.loadFromFile()

    def fixNS(self, ns, att):
        """Build the right tag or element name with namespace"""
        return "{" + Inkml.NS[ns] + "}" + att

    def loadFromFile(self):
        """load the ink from an inkml file (strokes, segments, labels)"""
        tree = ET.parse(self.fileName)
        # # ET.register_namespace();
        root = tree.getroot()
        for info in root.findall("ns:annotation", namespaces=Inkml.NS):
            if "type" in info.attrib:
                if info.attrib["type"] == "truth":
                    self.truth = info.text.strip()
                if info.attrib["type"] == "UI":
                    self.UI = info.text.strip()
        for strk in root.findall("ns:trace", namespaces=Inkml.NS):
            self.strokes[strk.attrib["id"]] = strk.text.strip()
            self.strkOrder.append(strk.attrib["id"])
        segments = root.find("ns:traceGroup", namespaces=Inkml.NS)
        if segments is None or len(segments) == 0:
            return
        for seg in segments.iterfind("ns:traceGroup", namespaces=Inkml.NS):
            id = seg.attrib[self.fixNS("xml", "id")]
            label = seg.find("ns:annotation", namespaces=Inkml.NS).text
            strkList = set([])
            for t in seg.findall("ns:traceView", namespaces=Inkml.NS):
                strkList.add(t.attrib["traceDataRef"])
            self.segments[id] = Segment(id, label, strkList)

    def getTraces(self, height=256):
        traces = []
        for trace_id in self.strkOrder:
            stroke_str = self.strokes[trace_id]
            point_strs = stroke_str.split(",")
            stroke = []
            for point_str in point_strs:
                parts = point_str.strip().split()
                if len(parts) >= 1:
                    x, y = float(parts[0]), float(parts[1])  # 只取前两个
                    stroke.append([x, y])
            traces.append(stroke)
        return traces

    def view(self):
        plt.figure(figsize=(16, 4))
        plt.axis("off")
        for trace in self.getTraces():
            trace_arr = np.array(trace)
            plt.plot(trace_arr[:, 0], -trace_arr[:, 1])  # invert y coordinate


def render_traces_to_image(traces, img_size=(768, 256), thickness=4, interp_points=128, invert=True, margin=16):
    W, H = img_size
    img = Image.new("L", img_size, color=255)
    # img = Image.new("RGB", img_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 对每条笔画做插值
    interp_traces = [interpolate_stroke(trace, interp_points) for trace in traces]

    interp_traces = [
        s for s in interp_traces
        if not np.allclose(s, s[0])
           or len(s) == 1  # 保留单点 trace（如省略号）
    ]

    # 将所有 trace 拼接成一个 array 计算 min/max
    all_points = np.vstack([np.array(t) for t in interp_traces])
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    # 归一化到 [0, 1]，再放大到图像大小
    norm_traces = [(np.array(t) - min_xy) / (max_xy - min_xy + 1e-8) for t in interp_traces]

    # 把归一化数据缩放到 drawable 区域
    drawable_size = np.array([W - 2 * margin, H - 2 * margin])
    scaled_traces = [t * drawable_size for t in norm_traces]

    # 计算内容实际 bbox 中心
    all_points = np.vstack(scaled_traces)
    content_center = (all_points.max(axis=0) + all_points.min(axis=0)) / 2

    # 将图像中心设置为 canvas 中心
    canvas_center = np.array([W / 2, H / 2])

    # 将内容中心移到画布中心
    centered_traces = [t - content_center + canvas_center for t in scaled_traces]

    for stroke in centered_traces:
        if len(stroke) >= 2:
            draw.line([tuple(p) for p in stroke], fill=0, width=thickness)
        elif len(stroke) == 1:
            x, y = stroke[0]
            r = thickness // 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=0)  # 绘制黑色圆点

    if invert:
        img = ImageOps.invert(img)

    # Resize + 抗锯齿
    img = img.resize(img_size, resample=Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.uint8)
