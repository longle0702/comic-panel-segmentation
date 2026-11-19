"""Line segment analysis (LSA) + geometric filtering (GF) post-processing.

This is a port of the reference notebook pipeline (``pipeline.py``,
``visualize_pipe``): every step, constant and quirk is reproduced so that the
output matches the notebook pixel for pixel. The only change is that the values
the notebook hardcoded for a 384x384 image are expressed relative to the actual
image size:

    notebook                         here
    dis_from_ref(384, 384)        -> dis_from_ref(w, h)
    border clip 0 <= coord <= 400 -> 0 <= x <= w * 400/384, 0 <= y <= h * 400/384

For a 384x384 mask the two are identical.

Steps (LSA):
    1. Opening with a 20x20 rectangle (smooth_edges).
    2. Canny(50, 100), then dilate 5x5 x2, erode 5x5, erode 3x3 (morphing).
    3. HoughLinesP(rho=2, theta=pi/180, threshold=25, minLineLength=10, maxLineGap=30).
    4. Split lines into horizontal / vertical (angle to the x axis >= 45 -> vertical).
    5. Group lines (group_line_and_outlier), merge groups that share a line and keep
       the lines at the modal angle, or the median-angle line (merge_lists_with_common_line).
    6. Intersections vertical x horizontal (inner) and with the image border.
    7. MeanShift(13) on inner and MeanShift(10) on border intersections -> centroids.
    8. Draw the merged lines (thickness 2) and carve a line (colour 0, thickness 3)
       into the smoothed mask between every pair of centroids that are > 40 px apart
       and whose Bresenham path is >= 75% covered by the drawn lines.

GF (not part of the notebook): keep connected components larger than 1000 px.
"""
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.cluster import MeanShift

# Size of the masks the notebook was written for; its hardcoded values refer to it.
NOTEBOOK_IMAGE_SIZE = 384.0
NOTEBOOK_BORDER_CLIP = 400.0


@dataclass(frozen=True)
class PostProcessParameters:
    opening_kernel_size: Tuple[int, int] = (20, 20)
    canny_thresh1: int = 50
    canny_thresh2: int = 100
    hough_rho: float = 2.0
    hough_theta: float = np.pi / 180.0
    hough_threshold: int = 25
    hough_min_line_length: float = 10.0
    hough_max_line_gap: float = 30.0
    # Grouping. The notebook's cross_angle rounds the cosine to -1/0/1 before acos,
    # so the 3 degree test effectively accepts lines up to ~60 degrees apart.
    angle_threshold_deg: float = 3.0
    distance_threshold_px: float = 8.5
    inner_bandwidth: float = 13.0
    border_bandwidth: float = 10.0
    # Border intersections are kept when 0 <= coord <= size * ratio (400/384 in the notebook).
    border_clip_ratio: float = NOTEBOOK_BORDER_CLIP / NOTEBOOK_IMAGE_SIZE
    # Connection check against the image of drawn merged lines.
    support_line_thickness: int = 2
    support_pixel_threshold: int = 20
    bresenham_support_threshold: float = 0.75
    min_euclidean_distance: float = 40.0
    line_color: int = 0
    line_thickness: int = 3
    # Geometric filtering.
    size_threshold: int = 1000


class DetectedLine:
    """Line segment; mirrors ``Line`` in the notebook."""

    def __init__(self, x1: int, y1: int, x2: int, y2: int, img_w: int = 384, img_h: int = 384):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.AB = (self.x2 - self.x1, self.y2 - self.y1)
        self.orientation = self._set_orientation()
        self.list_dis = self._dis_from_ref(img_w, img_h)

    def calculate_angle(self) -> float:
        ab = (self.x2 - self.x1, self.y2 - self.y1)
        mag_ab = np.sqrt(ab[0] ** 2 + ab[1] ** 2)
        dot_prod = np.dot(ab, (1, 0))
        angle = math.acos(dot_prod / mag_ab)
        return math.degrees(angle) % 360

    def _set_orientation(self) -> str:
        angle = self.calculate_angle()
        if angle > 180:
            angle = angle - 180
        if angle > 90:
            angle = 180 - angle
        return "vertical" if angle >= 45 else "horizontal"

    def cross_angle(self, other: "DetectedLine") -> float:
        ab = (self.x2 - self.x1, self.y2 - self.y1)
        cd = (other.x2 - other.x1, other.y2 - other.y1)
        mag_ab = np.sqrt(round(ab[0] ** 2 + ab[1] ** 2, 15))
        mag_cd = np.sqrt(round(cd[0] ** 2 + cd[1] ** 2, 15))
        dot_prod = round(np.dot(ab, cd), 15)
        mag_prod = round(mag_ab * mag_cd, 15)
        angle = math.acos(round(dot_prod / mag_prod))  # rounds to -1/0/1, as in the notebook
        return math.degrees(angle) % 360

    def _dis_from_ref(self, width: float, length: float) -> List[float]:
        ref_points = (
            (width * (5 / 7), length * 0.5),
            (width / 6, length * (6 / 7)),
            (width * (3.5 / 9), length * (1 / 3)),
        )
        ab = (self.x2 - self.x1, self.y2 - self.y1)
        mau = np.sqrt(ab[0] ** 2 + ab[1] ** 2)
        out = []
        for x00, y00 in ref_points:
            tu = abs((self.y2 - self.y1) * x00 - (self.x2 - self.x1) * y00 + self.x2 * self.y1 - self.x1 * self.y2)
            out.append(tu / mau)
        return out


def calc_inter(line1: DetectedLine, line2: DetectedLine) -> Optional[Tuple[int, int]]:
    """Intersection of the two infinite lines, truncated to int (notebook: calculate_intersection)."""
    x1, y1, x2, y2 = line1.x1, line1.y1, line1.x2, line1.y2
    x3, y3, x4, y4 = line2.x1, line2.y1, line2.x2, line2.y2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return int(px), int(py)


def bresenham_line(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    err = dx - dy
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points


def is_connection_valid(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    threshold: float = 0.75,
    pixel_threshold: int = 20,
) -> bool:
    """Share of the Bresenham path (out-of-image points included in the count) on drawn lines."""
    line_points = bresenham_line(x1, y1, x2, y2)
    colored_count = 0
    for x, y in line_points:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            if np.any(image[y, x] > pixel_threshold):
                colored_count += 1
    return (colored_count / len(line_points)) >= threshold


def filter_components_by_size(mask: np.ndarray, size_threshold: int = 1000) -> np.ndarray:
    bin_m = (mask > 127).astype(np.uint8) * 255
    num_labels, labels = cv2.connectedComponents(bin_m, connectivity=8)
    filtered = np.zeros_like(bin_m)
    for lbl in range(1, num_labels):
        comp_m = (labels == lbl).astype(np.uint8)
        if cv2.countNonZero(comp_m) > size_threshold:
            filtered[labels == lbl] = 255
    return filtered


def _group_lines(lines: Sequence[DetectedLine], params: PostProcessParameters) -> List[List[DetectedLine]]:
    """Notebook: group_line_and_outlier. Group i = line i + every later line close to it."""
    groups = []
    for i in range(len(lines)):
        group = [lines[i]]
        for j in range(i + 1, len(lines)):
            diff_angle = lines[i].cross_angle(lines[j])
            diff_dis = np.array(lines[i].list_dis) - np.array(lines[j].list_dis)
            angle_ok = (diff_angle < params.angle_threshold_deg) or (diff_angle > 180 - params.angle_threshold_deg)
            dis_ok = np.all(np.abs(diff_dis) < params.distance_threshold_px)
            if angle_ok and dis_ok:
                group.append(lines[j])
        groups.append(group)
    return groups


def _merge_groups(groups: List[List[DetectedLine]]) -> List[DetectedLine]:
    """Notebook: merge_lists_with_common_line.

    Groups sharing a line are merged. From each merged group, keep every line at the
    modal angle when that angle occurs at least twice, else every line at the median angle.
    The notebook's ``list(set(...))`` is replaced by an order-preserving de-duplication
    so results are deterministic; the kept lines are the same.
    """
    pending = [list(g) for g in groups]
    merged = []
    while pending:
        current = pending.pop(0)
        i = 0
        while i < len(pending):
            if any(line in pending[i] for line in current):
                current.extend(pending.pop(i))
            else:
                i += 1
        merged.append(list(dict.fromkeys(current)))

    final = []
    for group in merged:
        angles = {line: line.calculate_angle() for line in group}
        counts = Counter(angles.values())
        top = max(counts.values())
        mode_value = min(v for v, c in counts.items() if c == top)  # scipy.stats.mode picks the smallest
        if top >= 2:
            final.extend(k for k, v in angles.items() if v == mode_value)
        else:
            values = sorted(angles.values())
            median_value = values[len(values) // 2]
            final.extend(k for k, v in angles.items() if v == median_value)
    return final


def _mean_shift_centroids(points: List[Tuple[int, int]], bandwidth: float) -> np.ndarray:
    if not points:
        return np.empty((0, 2))
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(np.array(points))
    return ms.cluster_centers_


def run_line_segment_postprocessing(
    raw_mask: np.ndarray,
    params: Optional[PostProcessParameters] = None,
    apply_lsa: bool = True,
    apply_gf: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Post-process a raw panel mask (0/255 or 0/1, single channel).

    apply_lsa: line segment analysis (notebook pipeline).
    apply_gf:  geometric filtering (drop components of <= size_threshold px).
    Returns a 0/255 uint8 mask and diagnostics.
    """
    if params is None:
        params = PostProcessParameters()
    mask = np.asarray(raw_mask)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.uint8) * 255 if mask.max() <= 1 else mask.astype(np.uint8)
    h, w = mask.shape[:2]

    diagnostics: Dict[str, Any] = {
        "detected_lines_count": 0,
        "horizontal_merged_count": 0,
        "vertical_merged_count": 0,
        "centroids_count": 0,
        "validated_connections": 0,
    }

    if apply_lsa:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, params.opening_kernel_size)
        og_smoothed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        edges = cv2.Canny(og_smoothed, threshold1=params.canny_thresh1, threshold2=params.canny_thresh2)
        bridged = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        bridged = cv2.erode(bridged, np.ones((5, 5), np.uint8), iterations=1)
        bridged = cv2.erode(bridged, np.ones((3, 3), np.uint8), iterations=1)
        lines_raw = cv2.HoughLinesP(
            bridged,
            params.hough_rho,
            params.hough_theta,
            threshold=params.hough_threshold,
            minLineLength=params.hough_min_line_length,
            maxLineGap=params.hough_max_line_gap,
        )

        verticals: List[DetectedLine] = []
        horizontals: List[DetectedLine] = []
        if lines_raw is not None:
            for l in lines_raw:
                line = DetectedLine(*l[0], img_w=w, img_h=h)
                (verticals if line.orientation == "vertical" else horizontals).append(line)

        fin_h = _merge_groups(_group_lines(horizontals, params))
        fin_v = _merge_groups(_group_lines(verticals, params))

        # Image of the merged lines; connections are validated against it.
        support = np.zeros((h, w), np.uint8)
        for line in fin_h + fin_v:
            cv2.line(support, (line.x1, line.y1), (line.x2, line.y2), 255, params.support_line_thickness)

        border_lines = [
            DetectedLine(0, 0, w - 1, 0, img_w=w, img_h=h),
            DetectedLine(0, h - 1, w - 1, h - 1, img_w=w, img_h=h),
            DetectedLine(0, 0, 0, h - 1, img_w=w, img_h=h),
            DetectedLine(w - 1, 0, w - 1, h - 1, img_w=w, img_h=h),
        ]
        clip_x, clip_y = w * params.border_clip_ratio, h * params.border_clip_ratio

        def border_points(lines: List[DetectedLine]) -> List[Tuple[int, int]]:
            pts = [calc_inter(a, b) for a in lines for b in border_lines]
            return [p for p in pts if p is not None and 0 <= p[0] <= clip_x and 0 <= p[1] <= clip_y]

        inner = [p for p in (calc_inter(v, hl) for v in fin_v for hl in fin_h) if p is not None]
        border = border_points(fin_v) + border_points(fin_h)
        centroids = np.concatenate((
            _mean_shift_centroids(inner, params.inner_bandwidth),
            _mean_shift_centroids(border, params.border_bandwidth),
        ))

        carved = og_smoothed.copy()
        conns_drawn = 0
        for i, p1 in enumerate(centroids):
            for j, p2 in enumerate(centroids):
                if i == j:
                    continue
                x1, y1 = int(p1[0]), int(p1[1])
                x2, y2 = int(p2[0]), int(p2[1])
                if is_connection_valid(support, x1, y1, x2, y2, params.bresenham_support_threshold,
                                       params.support_pixel_threshold):
                    if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > params.min_euclidean_distance:
                        cv2.line(carved, (x1, y1), (x2, y2), params.line_color, params.line_thickness)
                        conns_drawn += 1
        out = carved
        diagnostics.update(
            detected_lines_count=0 if lines_raw is None else len(lines_raw),
            horizontal_merged_count=len(fin_h),
            vertical_merged_count=len(fin_v),
            centroids_count=len(centroids),
            validated_connections=conns_drawn,
        )
    else:
        out = mask

    if apply_gf:
        out = filter_components_by_size(out, size_threshold=params.size_threshold)
    return (out > 127).astype(np.uint8) * 255, diagnostics
