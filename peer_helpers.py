from __future__ import annotations

import hashlib
import inspect
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from peer_models import (
    CTVolume,
    DVHCurve,
    DoseVolume,
    ImageViewBounds,
    RTStructData,
    StructureGoal,
    StructureGoalEvaluation,
    StructureSliceContours,
)
from peer_dvh import (
    DEFAULT_DVH_OPTIONS,
    DVHCalculationOptions,
    _build_ct_slice_to_dose_transform,
    _build_local_occupancy_grid,
    _get_ct_geometry_context,
    _get_dose_sampling_context,
    _sample_dose_plane_virtual_rc,
    build_dvh_curve_from_weighted_samples,
    compute_oversampling_factor_from_metrics,
    compute_dvh_curves as compute_dvh_curves_high_accuracy_module,
    dose_at_volume_cc as dose_at_volume_cc_module,
    dose_at_volume_pct as dose_at_volume_pct_module,
    estimate_structure_geometry_metrics,
    _slice_thicknesses_mm,
    volume_cc_at_dose_gy as volume_cc_at_dose_gy_module,
    volume_pct_at_dose_gy as volume_pct_at_dose_gy_module,
)


HIGH_ACCURACY_DVH_OPTIONS = DVHCalculationOptions(
    automatic_oversampling=DEFAULT_DVH_OPTIONS.automatic_oversampling,
    fixed_oversampling_factor=DEFAULT_DVH_OPTIONS.fixed_oversampling_factor,
    use_fractional_labelmap=DEFAULT_DVH_OPTIONS.use_fractional_labelmap,
    fractional_subdivisions=DEFAULT_DVH_OPTIONS.fractional_subdivisions,
    use_linear_dose_interpolation=DEFAULT_DVH_OPTIONS.use_linear_dose_interpolation,
    # Use finer bins in the interactive app so small-structure D-metrics such
    # as D0.03cc do not collapse to the same value under coarse 0.2 Gy bins.
    dose_bin_width_gy=0.05,
    max_dose_gy=DEFAULT_DVH_OPTIONS.max_dose_gy,
    minimum_oversampling_factor=DEFAULT_DVH_OPTIONS.minimum_oversampling_factor,
    maximum_oversampling_factor=DEFAULT_DVH_OPTIONS.maximum_oversampling_factor,
    max_border_batch_points=DEFAULT_DVH_OPTIONS.max_border_batch_points,
)

SRS_SMALL_VOLUME_THRESHOLD_CC = 5.0

SRS_INTENSIVE_DVH_OPTIONS = DVHCalculationOptions(
    automatic_oversampling=False,
    fixed_oversampling_factor=8.0,
    use_fractional_labelmap=True,
    fractional_subdivisions=10,
    use_linear_dose_interpolation=True,
    dose_bin_width_gy=0.01,
    max_dose_gy=DEFAULT_DVH_OPTIONS.max_dose_gy,
    minimum_oversampling_factor=1.0,
    maximum_oversampling_factor=8.0,
    max_border_batch_points=max(DEFAULT_DVH_OPTIONS.max_border_batch_points, 600_000),
)


def safe_get(ds, name: str, default=None):
    return getattr(ds, name, default)


def normalize_structure_name(name: str) -> str:
    return " ".join(name.strip().upper().split())


def get_iop(ds) -> np.ndarray:
    return np.array(
        safe_get(ds, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        dtype=float,
    )


def get_ipp(ds) -> np.ndarray:
    return np.array(
        safe_get(ds, "ImagePositionPatient", [0.0, 0.0, 0.0]),
        dtype=float,
    )


def get_ct_row_col_normal(iop: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_cos = iop[:3]
    col_cos = iop[3:]
    normal = np.cross(row_cos, col_cos)

    row_norm = np.linalg.norm(row_cos)
    col_norm = np.linalg.norm(col_cos)
    normal_norm = np.linalg.norm(normal)

    if row_norm == 0 or col_norm == 0 or normal_norm == 0:
        raise ValueError("Invalid ImageOrientationPatient vectors.")

    row_cos = row_cos / row_norm
    col_cos = col_cos / col_norm
    normal = normal / normal_norm
    return row_cos, col_cos, normal


def patient_xyz_to_ct_rc(points_xyz: np.ndarray, ct: CTVolume, slice_index: int) -> np.ndarray:
    row_cos, col_cos, _ = get_ct_row_col_normal(ct.image_orientation_patient)
    slice_origin = np.asarray(ct.slice_origins_xyz_mm[slice_index], dtype=float)

    sx = float(ct.spacing_xyz_mm[0])
    sy = float(ct.spacing_xyz_mm[1])

    rel = points_xyz - slice_origin[None, :]
    cols = rel @ row_cos / sx
    rows = rel @ col_cos / sy
    return np.column_stack([rows, cols])


def nearest_slice_index(z_mm: float, z_positions_mm: np.ndarray) -> int:
    return int(np.argmin(np.abs(z_positions_mm - z_mm)))


def nearest_ct_slice_for_points(points_xyz: np.ndarray, ct: CTVolume) -> int:
    _, _, normal = get_ct_row_col_normal(ct.image_orientation_patient)
    contour_pos = float(np.mean(points_xyz @ normal))
    return int(np.argmin(np.abs(ct.z_positions_mm - contour_pos)))


def get_lowest_ptv_rx_gy(rtstruct: Optional[RTStructData]) -> Optional[float]:
    if rtstruct is None:
        return None

    lowest_rx_gy: Optional[float] = None
    for structure in rtstruct.structures:
        normalized_name = normalize_structure_name(structure.name)
        if not normalized_name.startswith("PTV"):
            continue

        digits = "".join(ch for ch in normalized_name if ch.isdigit())
        if not digits:
            continue

        rx_gy = float(int(digits)) / 100.0
        if lowest_rx_gy is None or rx_gy < lowest_rx_gy:
            lowest_rx_gy = rx_gy

    return lowest_rx_gy


def get_ptv_dose_levels_gy(rtstruct: Optional[RTStructData]) -> List[float]:
    if rtstruct is None:
        return []

    levels_gy: set[float] = set()
    for structure in rtstruct.structures:
        normalized_name = normalize_structure_name(structure.name)
        if not normalized_name.startswith("PTV"):
            continue

        digits = "".join(ch for ch in normalized_name if ch.isdigit())
        if not digits:
            continue

        rx_gy = float(int(digits)) / 100.0
        levels_gy.add(rx_gy)

    return sorted(levels_gy)


def compute_image_view_bounds(ct: CTVolume, hu_threshold: float = -900.0) -> Optional[ImageViewBounds]:
    mask = ct.volume_hu >= hu_threshold
    if not np.any(mask):
        return None

    axial_by_slice: Dict[int, Tuple[float, float, float, float]] = {}
    sx = float(ct.spacing_xyz_mm[0])
    sy = float(ct.spacing_xyz_mm[1])
    sz = float(ct.spacing_xyz_mm[2])
    sagittal_scale = orthogonal_row_scale(sz, sy)
    coronal_scale = orthogonal_row_scale(sz, sx)

    z_indices = np.where(np.any(mask, axis=(1, 2)))[0]
    y_indices = np.where(np.any(mask, axis=(0, 2)))[0]
    x_indices = np.where(np.any(mask, axis=(0, 1)))[0]

    for slice_index in z_indices.tolist():
        rows, cols = np.where(mask[slice_index])
        if rows.size == 0 or cols.size == 0:
            continue
        axial_by_slice[slice_index] = (
            float(np.min(cols)),
            float(np.max(cols)),
            float(np.min(rows)),
            float(np.max(rows)),
        )

    sagittal_bounds = None
    if y_indices.size > 0 and z_indices.size > 0:
        sagittal_bounds = (
            float(y_indices[0]),
            float(y_indices[-1]),
            float(z_indices[0] * sagittal_scale),
            float(z_indices[-1] * sagittal_scale),
        )

    coronal_bounds = None
    if x_indices.size > 0 and z_indices.size > 0:
        coronal_bounds = (
            float(x_indices[0]),
            float(x_indices[-1]),
            float(z_indices[0] * coronal_scale),
            float(z_indices[-1] * coronal_scale),
        )

    if not axial_by_slice and sagittal_bounds is None and coronal_bounds is None:
        return None

    return ImageViewBounds(
        axial_by_slice=axial_by_slice,
        sagittal=sagittal_bounds,
        coronal=coronal_bounds,
    )


def linear_resample_2d(src: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
    in_rows, in_cols = src.shape
    if in_rows == out_rows and in_cols == out_cols:
        return src.copy()

    row_coords = np.linspace(0, in_rows - 1, out_rows)
    col_coords = np.linspace(0, in_cols - 1, out_cols)

    r0 = np.floor(row_coords).astype(int)
    r1 = np.clip(r0 + 1, 0, in_rows - 1)
    dr = row_coords - r0

    c0 = np.floor(col_coords).astype(int)
    c1 = np.clip(c0 + 1, 0, in_cols - 1)
    dc = col_coords - c0

    out = np.empty((out_rows, out_cols), dtype=np.float32)
    for i in range(out_rows):
        top = (1.0 - dc) * src[r0[i], c0] + dc * src[r0[i], c1]
        bot = (1.0 - dc) * src[r1[i], c0] + dc * src[r1[i], c1]
        out[i, :] = (1.0 - dr[i]) * top + dr[i] * bot
    return out


def bilinear_sample_2d(src: np.ndarray, row_coords: np.ndarray, col_coords: np.ndarray) -> np.ndarray:
    out = np.zeros(row_coords.shape, dtype=np.float32)
    valid = (
        (row_coords >= 0.0)
        & (row_coords <= src.shape[0] - 1)
        & (col_coords >= 0.0)
        & (col_coords <= src.shape[1] - 1)
    )
    if not np.any(valid):
        return out

    rr = row_coords[valid]
    cc = col_coords[valid]

    r0 = np.floor(rr).astype(int)
    c0 = np.floor(cc).astype(int)
    r1 = np.clip(r0 + 1, 0, src.shape[0] - 1)
    c1 = np.clip(c0 + 1, 0, src.shape[1] - 1)

    dr = rr - r0
    dc = cc - c0

    top = (1.0 - dc) * src[r0, c0] + dc * src[r0, c1]
    bot = (1.0 - dc) * src[r1, c0] + dc * src[r1, c1]
    out[valid] = (1.0 - dr) * top + dr * bot
    return out


def resample_orthogonal_plane(plane: np.ndarray, row_spacing_mm: float, col_spacing_mm: float) -> np.ndarray:
    if plane.size == 0:
        return plane

    scale = row_spacing_mm / max(col_spacing_mm, 1e-6)
    out_rows = max(1, int(round(plane.shape[0] * scale)))
    if out_rows == plane.shape[0]:
        return plane
    return linear_resample_2d(plane, out_rows, plane.shape[1])


def orthogonal_row_scale(row_spacing_mm: float, col_spacing_mm: float) -> float:
    return row_spacing_mm / max(col_spacing_mm, 1e-6)


def line_intersections_at_col(contour_rc: np.ndarray, col_value: float) -> List[float]:
    intersections: List[float] = []
    if contour_rc.shape[0] < 2:
        return intersections

    for i in range(contour_rc.shape[0]):
        r0, c0 = contour_rc[i]
        r1, c1 = contour_rc[(i + 1) % contour_rc.shape[0]]
        if np.isclose(c0, c1):
            continue
        if (c0 <= col_value < c1) or (c1 <= col_value < c0):
            t = (col_value - c0) / (c1 - c0)
            intersections.append(float(r0 + t * (r1 - r0)))
    intersections.sort()
    return intersections


def line_intersections_at_row(contour_rc: np.ndarray, row_value: float) -> List[float]:
    intersections: List[float] = []
    if contour_rc.shape[0] < 2:
        return intersections

    for i in range(contour_rc.shape[0]):
        r0, c0 = contour_rc[i]
        r1, c1 = contour_rc[(i + 1) % contour_rc.shape[0]]
        if np.isclose(r0, r1):
            continue
        if (r0 <= row_value < r1) or (r1 <= row_value < r0):
            t = (row_value - r0) / (r1 - r0)
            intersections.append(float(c0 + t * (c1 - c0)))
    intersections.sort()
    return intersections


def build_outline_series(
    samples_by_slice: Dict[int, List[float]],
    row_scale: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    series: List[List[Tuple[float, float]]] = []
    for slice_index in sorted(samples_by_slice):
        intersections = samples_by_slice[slice_index]
        y = slice_index * row_scale
        while len(series) < len(intersections):
            series.append([])
        for idx, x in enumerate(intersections):
            series[idx].append((x, y))

    curves: List[Tuple[np.ndarray, np.ndarray]] = []
    for pts in series:
        if len(pts) < 2:
            continue
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        curves.append((xs, ys))
    return curves


def sample_dose_to_ct_slice(ct: CTVolume, dose: DoseVolume, ct_slice_index: int) -> np.ndarray:
    ct_origin = np.asarray(ct.slice_origins_xyz_mm[ct_slice_index], dtype=float)
    ct_col_cos, ct_row_cos, _ = get_ct_row_col_normal(ct.image_orientation_patient)
    dose_col_cos, dose_row_cos, dose_normal = get_ct_row_col_normal(dose.image_orientation_patient)

    dose_plane_pos_mm = float(np.dot(ct_origin, dose_normal))
    dose_k = nearest_slice_index(dose_plane_pos_mm, dose.z_positions_mm)
    dose_plane = dose.dose_gy[dose_k]

    ct_sx = float(ct.spacing_xyz_mm[0])
    ct_sy = float(ct.spacing_xyz_mm[1])
    dose_sx = float(dose.spacing_xyz_mm[0])
    dose_sy = float(dose.spacing_xyz_mm[1])

    row_coords = np.arange(ct.rows, dtype=np.float32)[:, None]
    col_coords = np.arange(ct.cols, dtype=np.float32)[None, :]
    patient_xyz = (
        ct_origin[None, None, :]
        + col_coords[:, :, None] * ct_sx * ct_col_cos[None, None, :]
        + row_coords[:, :, None] * ct_sy * ct_row_cos[None, None, :]
    )

    rel = patient_xyz - dose.slice_origins_xyz_mm[dose_k][None, None, :]
    dose_col_coords = np.tensordot(rel, dose_col_cos, axes=([2], [0])) / dose_sx
    dose_row_coords = np.tensordot(rel, dose_row_cos, axes=([2], [0])) / dose_sy
    return bilinear_sample_2d(dose_plane, dose_row_coords, dose_col_coords)


def dose_to_rgba(
    dose_plane_gy: np.ndarray,
    alpha: float = 0.35,
    min_dose_gy: Optional[float] = None,
    max_dose_gy: Optional[float] = None,
) -> np.ndarray:
    if np.nanmax(dose_plane_gy) <= 0:
        return np.zeros(dose_plane_gy.shape + (4,), dtype=np.uint8)

    if min_dose_gy is None:
        min_dose_gy = 0.0
    if max_dose_gy is None:
        max_dose_gy = float(np.nanmax(dose_plane_gy))

    min_dose_gy = float(min_dose_gy)
    max_dose_gy = max(float(max_dose_gy), min_dose_gy + 1e-6)

    clipped = np.clip(dose_plane_gy, min_dose_gy, max_dose_gy)
    norm = (clipped - min_dose_gy) / (max_dose_gy - min_dose_gy)

    r = np.clip(3.0 * norm, 0.0, 1.0)
    g = np.clip(3.0 * norm - 1.0, 0.0, 1.0)
    b = np.clip(3.0 * norm - 2.0, 0.0, 1.0)

    visible_mask = dose_plane_gy > min_dose_gy
    a = visible_mask.astype(np.float32) * alpha

    rgba = np.stack([r, g, b, a], axis=-1)
    return (rgba * 255).astype(np.uint8)


def rasterize_polygon_mask(contour_rc: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if contour_rc.shape[0] < 3:
        return np.zeros((rows, cols), dtype=bool)

    rr = contour_rc[:, 0]
    cc = contour_rc[:, 1]

    r_min = max(int(np.floor(np.min(rr))), 0)
    r_max = min(int(np.ceil(np.max(rr))), rows - 1)
    c_min = max(int(np.floor(np.min(cc))), 0)
    c_max = min(int(np.ceil(np.max(cc))), cols - 1)
    if r_min > r_max or c_min > c_max:
        return np.zeros((rows, cols), dtype=bool)

    sample_rows = np.arange(r_min, r_max + 1, dtype=np.float32)[:, None] + 0.5
    sample_cols = np.arange(c_min, c_max + 1, dtype=np.float32)[None, :] + 0.5

    inside = np.zeros((r_max - r_min + 1, c_max - c_min + 1), dtype=bool)
    next_rr = np.roll(rr, -1)
    next_cc = np.roll(cc, -1)

    for r0, c0, r1, c1 in zip(rr, cc, next_rr, next_cc):
        if np.isclose(r0, r1):
            continue
        intersects = ((r0 > sample_rows) != (r1 > sample_rows))
        cross_cols = (c1 - c0) * (sample_rows - r0) / (r1 - r0) + c0
        inside ^= intersects & (sample_cols < cross_cols)

    mask = np.zeros((rows, cols), dtype=bool)
    mask[r_min:r_max + 1, c_min:c_max + 1] = inside
    return mask


def build_structure_slice_mask(structure: StructureSliceContours, slice_index: int, rows: int, cols: int) -> np.ndarray:
    mask = np.zeros((rows, cols), dtype=bool)
    for contour_rc in structure.points_rc_by_slice.get(slice_index, []):
        mask ^= rasterize_polygon_mask(contour_rc, rows, cols)
    return mask


def build_structure_mask_cache(
    rtstruct: RTStructData,
    rows: int,
    cols: int,
) -> List[Dict[int, np.ndarray]]:
    mask_cache: List[Dict[int, np.ndarray]] = []
    for structure in rtstruct.structures:
        structure_masks: Dict[int, np.ndarray] = {}
        for slice_index in structure.points_rc_by_slice:
            structure_masks[slice_index] = build_structure_slice_mask(structure, slice_index, rows, cols)
        mask_cache.append(structure_masks)
    return mask_cache


def compute_cumulative_dvh(dose_values_gy: np.ndarray, max_dose_gy: float, num_bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    if dose_values_gy.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    max_dose_gy = max(float(max_dose_gy), float(np.nanmax(dose_values_gy)), 1e-3)
    bin_edges = np.linspace(0.0, max_dose_gy, num_bins + 1, dtype=np.float32)
    hist, _ = np.histogram(np.clip(dose_values_gy, 0.0, max_dose_gy), bins=bin_edges)
    cumulative = np.cumsum(hist[::-1])[::-1].astype(np.float32)
    cumulative *= 100.0 / max(float(dose_values_gy.size), 1.0)
    dose_axis = np.append(bin_edges[:-1], bin_edges[-1])
    volume_axis = np.append(cumulative, 0.0)
    return dose_axis, volume_axis


def parse_goal_value(value_text: str) -> Tuple[Optional[float], str]:
    text = value_text.strip()
    if not text:
        return None, ""

    match = re.match(r"^\s*([<>]=?|=)?\s*([-+]?\d*\.?\d+)\s*([A-Za-z%]+)?\s*$", text)
    if match is None:
        return None, text

    value = float(match.group(2))
    unit = (match.group(3) or "").upper()
    if unit == "CM3":
        unit = "CC"
    return value, unit


def format_metric_value(value: Optional[float], unit: str) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    normalized_unit = unit.upper()
    if normalized_unit == "%":
        return f"{value:.1f}%"
    if normalized_unit == "CC":
        return f"{value:.2f} cc"
    if normalized_unit == "GY":
        return f"{value:.2f} Gy"
    return f"{value:.2f} {unit}".strip()


def volume_pct_at_dose_gy(curve: DVHCurve, dose_gy: float) -> float:
    return volume_pct_at_dose_gy_module(curve, dose_gy)


def volume_cc_at_dose_gy(curve: DVHCurve, dose_gy: float) -> float:
    return volume_cc_at_dose_gy_module(curve, dose_gy)


def dose_at_volume_pct(curve: DVHCurve, volume_pct: float) -> float:
    return dose_at_volume_pct_module(curve, volume_pct)


def dose_at_volume_cc(curve: DVHCurve, volume_cc: float) -> float:
    return dose_at_volume_cc_module(curve, volume_cc)


def parse_v_metric_threshold_gy(metric_key: str) -> Optional[float]:
    match = re.fullmatch(r"V([-+]?\d*\.?\d+)(?:GY)?", metric_key)
    if match is None:
        return None
    return float(match.group(1))


def parse_d_metric_volume(metric_key: str) -> Tuple[Optional[float], str]:
    match = re.fullmatch(r"D([-+]?\d*\.?\d+)(CC|%)", metric_key)
    if match is None:
        return None, ""
    return float(match.group(1)), match.group(2)


def evaluate_structure_goal(curve: DVHCurve, goal: StructureGoal) -> StructureGoalEvaluation:
    metric_key = goal.metric.strip().upper().replace(" ", "")
    goal_value, goal_unit = parse_goal_value(goal.value_text)
    actual_value: Optional[float] = None
    actual_unit = "GY"

    if metric_key in {"MEAN", "DMEAN"}:
        actual_value = curve.mean_dose_gy
        actual_unit = "GY"
    elif metric_key in {"MAX", "DMAX"}:
        actual_value = curve.max_dose_gy
        actual_unit = "GY"
    elif metric_key in {"MIN", "DMIN"}:
        actual_value = curve.min_dose_gy
        actual_unit = "GY"
    else:
        dose_threshold_gy = parse_v_metric_threshold_gy(metric_key)
        if dose_threshold_gy is not None:
            if goal_unit == "%":
                actual_value = volume_pct_at_dose_gy(curve, dose_threshold_gy)
                actual_unit = "%"
            else:
                actual_value = volume_cc_at_dose_gy(curve, dose_threshold_gy)
                actual_unit = "CC"
        else:
            volume_target, volume_unit = parse_d_metric_volume(metric_key)
            if volume_target is not None:
                if volume_unit == "%":
                    actual_value = dose_at_volume_pct(curve, volume_target)
                    actual_unit = "GY"
                elif curve.volume_cc > 0.0:
                    actual_value = dose_at_volume_pct(curve, 100.0 * volume_target / curve.volume_cc)
                    actual_unit = "GY"

    passed: Optional[bool] = None
    if actual_value is not None and goal_value is not None:
        comparator = goal.comparator.strip()
        if comparator in {"<", "<="}:
            passed = actual_value <= goal_value if comparator == "<=" else actual_value < goal_value
        elif comparator in {">", ">="}:
            passed = actual_value >= goal_value if comparator == ">=" else actual_value > goal_value
        elif comparator in {"=", "=="}:
            passed = bool(np.isclose(actual_value, goal_value))

    return StructureGoalEvaluation(
        metric=goal.metric,
        comparator=goal.comparator,
        goal_text=goal.value_text,
        actual_text=format_metric_value(actual_value, actual_unit),
        passed=passed,
    )


def evaluate_structure_goals(
    curves: List[DVHCurve],
    goals_by_structure: Dict[str, List[StructureGoal]],
) -> Dict[str, List[StructureGoalEvaluation]]:
    curve_by_name = {normalize_structure_name(curve.name): curve for curve in curves}
    evaluations: Dict[str, List[StructureGoalEvaluation]] = {}
    for structure_name, goals in goals_by_structure.items():
        curve = curve_by_name.get(structure_name)
        if curve is None:
            continue
        evaluations[structure_name] = [evaluate_structure_goal(curve, goal) for goal in goals]
    return evaluations


def _subset_mask_cache(
    mask_cache: Optional[List[Dict[int, np.ndarray]]],
    indices: List[int],
) -> Optional[List[Dict[int, np.ndarray]]]:
    if mask_cache is None:
        return None
    if any(index < 0 or index >= len(mask_cache) for index in indices):
        return None
    return [mask_cache[index] for index in indices]


def _should_use_srs_intensive_options(
    ct: CTVolume,
    dose: DoseVolume,
    structure: StructureSliceContours,
    structure_mask_cache: Optional[Dict[int, np.ndarray]] = None,
) -> bool:
    metrics = estimate_structure_geometry_metrics(
        ct,
        dose,
        structure,
        structure_mask_cache=structure_mask_cache,
    )
    return (metrics.volume_mm3 / 1000.0) <= SRS_SMALL_VOLUME_THRESHOLD_CC


def compute_single_structure_high_accuracy_curve(
    ct: CTVolume,
    dose: DoseVolume,
    structure: StructureSliceContours,
    structure_mask_cache: Optional[Dict[int, np.ndarray]] = None,
    options: Optional[DVHCalculationOptions] = None,
) -> Optional[DVHCurve]:
    if options is None:
        options = (
            SRS_INTENSIVE_DVH_OPTIONS
            if _should_use_srs_intensive_options(ct, dose, structure, structure_mask_cache)
            else HIGH_ACCURACY_DVH_OPTIONS
        )
    if _should_use_srs_intensive_options(ct, dose, structure, structure_mask_cache):
        return _compute_srs_interpolated_curve(
            ct,
            dose,
            structure,
            structure_mask_cache=structure_mask_cache,
            options=options,
        )
    rtstruct = RTStructData(
        structures=[structure],
        frame_of_reference_uid=getattr(ct, "frame_of_reference_uid", ""),
    )
    mask_cache = [structure_mask_cache] if structure_mask_cache is not None else None
    curves = compute_dvh_curves_high_accuracy_module(
        ct,
        dose,
        rtstruct,
        options=options,
        mask_cache=mask_cache,
    )
    return curves[0] if curves else None


def _compute_srs_sampling_scale(
    ct: CTVolume,
    dose: DoseVolume,
    structure: StructureSliceContours,
    structure_mask_cache: Optional[Dict[int, np.ndarray]],
    options: DVHCalculationOptions,
) -> float:
    metrics = estimate_structure_geometry_metrics(
        ct,
        dose,
        structure,
        structure_mask_cache=structure_mask_cache,
    )
    decision = (
        compute_oversampling_factor_from_metrics(metrics.rss, metrics.complexity, options)
        if options.automatic_oversampling
        else None
    )
    scale = (
        float(decision.oversampling_factor)
        if decision is not None
        else float(options.fixed_oversampling_factor)
    )
    return float(np.clip(scale, options.minimum_oversampling_factor, options.maximum_oversampling_factor))


def _minimum_positive_step(values: np.ndarray) -> Optional[float]:
    if values.size < 2:
        return None
    sorted_values = np.sort(np.unique(values.astype(np.float64, copy=False)))
    diffs = np.diff(sorted_values)
    positive = diffs[diffs > 1e-6]
    if positive.size == 0:
        return None
    return float(np.min(positive))


def _contour_signed_area_rc(contour_rc: np.ndarray) -> float:
    contour = np.asarray(contour_rc, dtype=np.float64)
    if contour.ndim != 2 or contour.shape[0] < 3:
        return 0.0
    cols = contour[:, 1]
    rows = contour[:, 0]
    return 0.5 * float(
        np.sum(cols * np.roll(rows, -1) - np.roll(cols, -1) * rows)
    )


def _resample_contour_rc(contour_rc: np.ndarray, point_count: int) -> np.ndarray:
    contour = np.asarray(contour_rc, dtype=np.float32)
    if contour.shape[0] == 0 or point_count <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if contour.shape[0] == 1:
        return np.repeat(contour.astype(np.float32, copy=False), point_count, axis=0)

    closed = np.vstack([contour, contour[0]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length <= 1e-6:
        return np.repeat(contour[:1].astype(np.float32, copy=False), point_count, axis=0)

    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    sample_positions = np.linspace(0.0, total_length, point_count, endpoint=False, dtype=np.float64)
    resampled = np.zeros((point_count, 2), dtype=np.float32)
    segment_index = 0
    for sample_index, sample_position in enumerate(sample_positions):
        while segment_index < segment_lengths.size - 1 and sample_position > cumulative[segment_index + 1]:
            segment_index += 1
        start_point = closed[segment_index]
        end_point = closed[segment_index + 1]
        start_distance = cumulative[segment_index]
        segment_length = max(float(segment_lengths[segment_index]), 1e-6)
        alpha = np.float32((sample_position - start_distance) / segment_length)
        resampled[sample_index] = start_point * np.float32(1.0 - alpha) + end_point * alpha
    return resampled


def _align_resampled_ring_rc(reference_ring: np.ndarray, ring: np.ndarray) -> np.ndarray:
    if reference_ring.shape != ring.shape or ring.shape[0] == 0:
        return ring.astype(np.float32, copy=False)
    best_shift = 0
    best_score = np.inf
    for shift in range(ring.shape[0]):
        rolled = np.roll(ring, shift, axis=0)
        score = float(np.sum((reference_ring - rolled) ** 2))
        if score < best_score:
            best_score = score
            best_shift = shift
    return np.roll(ring, best_shift, axis=0).astype(np.float32, copy=False)


def _build_srs_interpolated_occupancy_model(
    ct: CTVolume,
    dose: DoseVolume,
    structure: StructureSliceContours,
    *,
    structure_mask_cache: Optional[Dict[int, np.ndarray]] = None,
    options: DVHCalculationOptions = SRS_INTENSIVE_DVH_OPTIONS,
) -> Optional[Dict[str, object]]:
    if not structure.points_rc_by_slice:
        return None

    scale = _compute_srs_sampling_scale(ct, dose, structure, structure_mask_cache, options)
    ct_context = _get_ct_geometry_context(ct)
    dose_context = _get_dose_sampling_context(dose)
    slice_thicknesses = _slice_thicknesses_mm(ct.z_positions_mm, float(ct.spacing_xyz_mm[2]))

    payloads: List[Dict[str, object]] = []
    row_min: Optional[int] = None
    row_max: Optional[int] = None
    col_min: Optional[int] = None
    col_max: Optional[int] = None

    for slice_index in sorted(structure.points_rc_by_slice):
        contours_rc = structure.points_rc_by_slice.get(slice_index, [])
        if not contours_rc:
            continue
        slice_transform = _build_ct_slice_to_dose_transform(ct_context, dose_context, slice_index)
        contours_dose_rc = [
            (
                np.asarray(contour, dtype=np.float32) @ slice_transform.transform_rc_to_dose_rc.T
                + slice_transform.offset_rc_to_dose_rc[None, :]
            ).astype(np.float32, copy=False)
            for contour in contours_rc
        ]
        occupancy, row_offset, col_offset = _build_local_occupancy_grid(
            contours_dose_rc,
            scale,
            use_fractional_labelmap=options.use_fractional_labelmap,
            fractional_subdivisions=options.fractional_subdivisions,
            max_border_batch_points=options.max_border_batch_points,
        )
        if occupancy.size == 0 or not np.any(occupancy > 0.0):
            continue

        local_row_min = int(row_offset)
        local_row_max = int(row_offset + occupancy.shape[0] - 1)
        local_col_min = int(col_offset)
        local_col_max = int(col_offset + occupancy.shape[1] - 1)
        row_min = local_row_min if row_min is None else min(row_min, local_row_min)
        row_max = local_row_max if row_max is None else max(row_max, local_row_max)
        col_min = local_col_min if col_min is None else min(col_min, local_col_min)
        col_max = local_col_max if col_max is None else max(col_max, local_col_max)
        payloads.append(
            {
                "slice_index": int(slice_index),
                "z_mm": float(slice_transform.plane_position_mm),
                "slab_thickness_mm": (
                    float(slice_thicknesses[slice_index])
                    if slice_index < slice_thicknesses.size
                    else float(ct.spacing_xyz_mm[2])
                ),
                "row_offset": int(row_offset),
                "col_offset": int(col_offset),
                "occupancy": occupancy.astype(np.float32, copy=False),
                "contours_dose_rc": contours_dose_rc,
            }
        )

    if not payloads or row_min is None or row_max is None or col_min is None or col_max is None:
        return None

    payloads.sort(key=lambda payload: float(payload["z_mm"]))
    height = int(row_max - row_min + 1)
    width = int(col_max - col_min + 1)
    anchor_planes: List[np.ndarray] = []
    anchor_z_mm = np.array([float(payload["z_mm"]) for payload in payloads], dtype=np.float64)
    slab_thicknesses_mm = np.array([float(payload["slab_thickness_mm"]) for payload in payloads], dtype=np.float64)

    for payload in payloads:
        embedded = np.zeros((height, width), dtype=np.float32)
        occupancy = np.asarray(payload["occupancy"], dtype=np.float32)
        rr0 = int(payload["row_offset"]) - row_min
        cc0 = int(payload["col_offset"]) - col_min
        rr1 = rr0 + occupancy.shape[0]
        cc1 = cc0 + occupancy.shape[1]
        embedded[rr0:rr1, cc0:cc1] = occupancy
        anchor_planes.append(embedded)

    z_spacing_candidates = [
        value
        for value in [
            _minimum_positive_step(dose_context.z_positions_mm),
            _minimum_positive_step(anchor_z_mm),
            _minimum_positive_step(ct.z_positions_mm),
            float(ct.spacing_xyz_mm[2]),
        ]
        if value is not None and value > 1e-6
    ]
    base_z_spacing_mm = min(z_spacing_candidates) if z_spacing_candidates else max(float(ct.spacing_xyz_mm[2]), 1.0)
    z_step_mm = max(base_z_spacing_mm / scale, 0.05)

    z_min_mm = float(anchor_z_mm[0] - slab_thicknesses_mm[0] / 2.0)
    z_max_mm = float(anchor_z_mm[-1] + slab_thicknesses_mm[-1] / 2.0)
    if z_max_mm <= z_min_mm:
        z_max_mm = z_min_mm + z_step_mm

    z_count = max(1, int(np.ceil((z_max_mm - z_min_mm) / z_step_mm)))
    z_step_mm = float((z_max_mm - z_min_mm) / z_count)
    z_samples_mm = z_min_mm + (np.arange(z_count, dtype=np.float64) + 0.5) * z_step_mm

    occupancy_stack = np.zeros((z_count, height, width), dtype=np.float32)

    use_mesh_interpolation = all(
        len(payload.get("contours_dose_rc", [])) == 1
        for payload in payloads
    )
    if use_mesh_interpolation:
        point_count = max(64, max(len(np.asarray(payload["contours_dose_rc"][0])) for payload in payloads))
        resampled_rings: List[np.ndarray] = []
        for payload in payloads:
            ring = _resample_contour_rc(np.asarray(payload["contours_dose_rc"][0], dtype=np.float32), point_count)
            if _contour_signed_area_rc(ring) < 0.0:
                ring = ring[::-1].copy()
            if resampled_rings:
                ring = _align_resampled_ring_rc(resampled_rings[-1], ring)
            resampled_rings.append(ring.astype(np.float32, copy=False))

        extended_rings: List[np.ndarray] = []
        extended_z_mm: List[float] = []
        first_ring = resampled_rings[0]
        last_ring = resampled_rings[-1]
        first_centroid = np.mean(first_ring, axis=0, dtype=np.float32)
        last_centroid = np.mean(last_ring, axis=0, dtype=np.float32)
        first_cap_ring = np.repeat(first_centroid[None, :], point_count, axis=0).astype(np.float32, copy=False)
        last_cap_ring = np.repeat(last_centroid[None, :], point_count, axis=0).astype(np.float32, copy=False)
        first_cap_z = float(anchor_z_mm[0] - slab_thicknesses_mm[0] / 2.0)
        last_cap_z = float(anchor_z_mm[-1] + slab_thicknesses_mm[-1] / 2.0)

        extended_rings.append(first_cap_ring)
        extended_z_mm.append(first_cap_z)
        extended_rings.extend(resampled_rings)
        extended_z_mm.extend(anchor_z_mm.tolist())
        extended_rings.append(last_cap_ring)
        extended_z_mm.append(last_cap_z)

        for z_index, z_mm in enumerate(z_samples_mm):
            if z_mm <= extended_z_mm[0]:
                interp_ring = extended_rings[0]
            elif z_mm >= extended_z_mm[-1]:
                interp_ring = extended_rings[-1]
            else:
                upper_index = int(np.searchsorted(np.asarray(extended_z_mm, dtype=np.float64), z_mm, side="right"))
                lower_index = max(0, upper_index - 1)
                z0 = float(extended_z_mm[lower_index])
                z1 = float(extended_z_mm[min(upper_index, len(extended_z_mm) - 1)])
                if z1 <= z0 + 1e-6:
                    interp_ring = extended_rings[lower_index]
                else:
                    alpha = np.float32((z_mm - z0) / (z1 - z0))
                    interp_ring = (
                        extended_rings[lower_index] * np.float32(1.0 - alpha)
                        + extended_rings[min(upper_index, len(extended_rings) - 1)] * alpha
                    ).astype(np.float32, copy=False)

            occupancy, row_offset, col_offset = _build_local_occupancy_grid(
                [interp_ring],
                scale,
                use_fractional_labelmap=options.use_fractional_labelmap,
                fractional_subdivisions=options.fractional_subdivisions,
                max_border_batch_points=options.max_border_batch_points,
            )
            if occupancy.size == 0 or not np.any(occupancy > 0.0):
                continue
            rr0 = int(row_offset) - row_min
            cc0 = int(col_offset) - col_min
            rr1 = rr0 + occupancy.shape[0]
            cc1 = cc0 + occupancy.shape[1]
            occupancy_stack[z_index, rr0:rr1, cc0:cc1] = np.maximum(
                occupancy_stack[z_index, rr0:rr1, cc0:cc1],
                occupancy.astype(np.float32, copy=False),
            )
    else:
        try:
            from scipy.ndimage import binary_fill_holes, distance_transform_edt  # type: ignore
        except Exception:  # pragma: no cover - optional runtime dependency
            binary_fill_holes = None
            distance_transform_edt = None

        if distance_transform_edt is not None:
            sampling_2d = (
                float(dose_context.spacing_row_mm / scale),
                float(dose_context.spacing_col_mm / scale),
            )
            binary_anchor_planes: List[np.ndarray] = []
            signed_distance_planes: List[np.ndarray] = []
            for plane in anchor_planes:
                binary_plane = np.asarray(plane > 1e-6, dtype=bool)
                if binary_fill_holes is not None and np.any(binary_plane):
                    binary_plane = np.asarray(binary_fill_holes(binary_plane), dtype=bool)
                binary_anchor_planes.append(binary_plane)
                signed_distance_planes.append(
                    distance_transform_edt(~binary_plane, sampling=sampling_2d)
                    - distance_transform_edt(binary_plane, sampling=sampling_2d)
                )

            if anchor_z_mm.size == 1:
                occupancy_stack[:] = binary_anchor_planes[0].astype(np.float32)[None, :, :]
            else:
                for z_index, z_mm in enumerate(z_samples_mm):
                    if z_mm <= anchor_z_mm[0]:
                        occupancy_stack[z_index] = binary_anchor_planes[0].astype(np.float32, copy=False)
                        continue
                    if z_mm >= anchor_z_mm[-1]:
                        occupancy_stack[z_index] = binary_anchor_planes[-1].astype(np.float32, copy=False)
                        continue
                    upper_index = int(np.searchsorted(anchor_z_mm, z_mm, side="right"))
                    lower_index = max(0, upper_index - 1)
                    z0 = float(anchor_z_mm[lower_index])
                    z1 = float(anchor_z_mm[min(upper_index, anchor_z_mm.size - 1)])
                    if z1 <= z0 + 1e-6:
                        occupancy_stack[z_index] = binary_anchor_planes[lower_index].astype(np.float32, copy=False)
                        continue
                    alpha = float((z_mm - z0) / (z1 - z0))
                    sdf = (
                        signed_distance_planes[lower_index] * (1.0 - alpha)
                        + signed_distance_planes[min(upper_index, anchor_z_mm.size - 1)] * alpha
                    )
                    occupancy_stack[z_index] = np.asarray(sdf <= 0.0, dtype=np.float32)
        else:
            if anchor_z_mm.size == 1:
                occupancy_stack[:] = anchor_planes[0][None, :, :]
            else:
                for z_index, z_mm in enumerate(z_samples_mm):
                    if z_mm <= anchor_z_mm[0]:
                        occupancy_stack[z_index] = anchor_planes[0]
                        continue
                    if z_mm >= anchor_z_mm[-1]:
                        occupancy_stack[z_index] = anchor_planes[-1]
                        continue
                    upper_index = int(np.searchsorted(anchor_z_mm, z_mm, side="right"))
                    lower_index = max(0, upper_index - 1)
                    z0 = float(anchor_z_mm[lower_index])
                    z1 = float(anchor_z_mm[min(upper_index, anchor_z_mm.size - 1)])
                    if z1 <= z0 + 1e-6:
                        occupancy_stack[z_index] = anchor_planes[lower_index]
                        continue
                    alpha = np.float32((z_mm - z0) / (z1 - z0))
                    occupancy_stack[z_index] = (
                        anchor_planes[lower_index] * np.float32(1.0 - alpha)
                        + anchor_planes[min(upper_index, anchor_z_mm.size - 1)] * alpha
                    )

    local_rows = np.arange(height, dtype=np.float32)
    local_cols = np.arange(width, dtype=np.float32)
    row_grid, col_grid = np.meshgrid(local_rows, local_cols, indexing="ij")
    row_coords_rc = ((np.float32(row_min) + row_grid + 0.5) / np.float32(scale)).astype(np.float32, copy=False)
    col_coords_rc = ((np.float32(col_min) + col_grid + 0.5) / np.float32(scale)).astype(np.float32, copy=False)

    return {
        "occupancy_stack": occupancy_stack,
        "z_samples_mm": z_samples_mm.astype(np.float32),
        "z_step_mm": float(z_step_mm),
        "row_coords_rc": row_coords_rc,
        "col_coords_rc": col_coords_rc,
        "cell_area_mm2": float(dose_context.spacing_row_mm * dose_context.spacing_col_mm / (scale * scale)),
        "row_spacing_mm": float(dose_context.spacing_row_mm / scale),
        "col_spacing_mm": float(dose_context.spacing_col_mm / scale),
        "oversampling_factor": float(scale),
        "used_fractional_labelmap": bool(options.use_fractional_labelmap),
    }


def _compute_srs_interpolated_curve(
    ct: CTVolume,
    dose: DoseVolume,
    structure: StructureSliceContours,
    *,
    structure_mask_cache: Optional[Dict[int, np.ndarray]] = None,
    options: DVHCalculationOptions = SRS_INTENSIVE_DVH_OPTIONS,
) -> Optional[DVHCurve]:
    model = _build_srs_interpolated_occupancy_model(
        ct,
        dose,
        structure,
        structure_mask_cache=structure_mask_cache,
        options=options,
    )
    if model is None:
        return None

    occupancy_stack = np.asarray(model["occupancy_stack"], dtype=np.float32)
    z_samples_mm = np.asarray(model["z_samples_mm"], dtype=np.float32)
    z_step_mm = float(model["z_step_mm"])
    row_coords_rc = np.asarray(model["row_coords_rc"], dtype=np.float32)
    col_coords_rc = np.asarray(model["col_coords_rc"], dtype=np.float32)
    cell_area_mm2 = float(model["cell_area_mm2"])

    all_dose_values: List[np.ndarray] = []
    all_weights_cc: List[np.ndarray] = []
    for z_index, z_mm in enumerate(z_samples_mm):
        occupancy_plane = occupancy_stack[z_index]
        active_mask = occupancy_plane > 1e-6
        if not np.any(active_mask):
            continue
        weights_cc = (
            occupancy_plane[active_mask].astype(np.float64, copy=False)
            * cell_area_mm2
            * z_step_mm
            / 1000.0
        )
        dose_values = _sample_dose_plane_virtual_rc(
            dose,
            row_coords_rc[active_mask],
            col_coords_rc[active_mask],
            float(z_mm),
            linear_interpolation=options.use_linear_dose_interpolation,
            dose_context=_get_dose_sampling_context(dose),
        ).astype(np.float64, copy=False)
        all_dose_values.append(dose_values)
        all_weights_cc.append(weights_cc)

    if not all_dose_values or not all_weights_cc:
        return None

    dose_values = np.concatenate(all_dose_values)
    weights_cc = np.concatenate(all_weights_cc)
    curve = build_dvh_curve_from_weighted_samples(
        structure.name,
        structure.color_rgb,
        dose_values,
        weights_cc,
        bin_width_gy=options.dose_bin_width_gy,
        max_dose_gy=options.max_dose_gy if options.max_dose_gy is not None else float(np.nanmax(dose.dose_gy)),
        oversampling_factor=float(model["oversampling_factor"]),
        used_fractional_labelmap=bool(model["used_fractional_labelmap"]),
        metadata={
            "interpolated_3d_small_structure": "true",
            "volume_mm3": float(np.sum(weights_cc) * 1000.0),
        },
    )
    return curve


def fill_binary_holes_2d(mask_2d: np.ndarray) -> np.ndarray:
    filled = np.asarray(mask_2d, dtype=bool).copy()
    background = ~filled
    if not np.any(background):
        return filled

    visited = np.zeros_like(filled, dtype=bool)
    stack: List[Tuple[int, int]] = []
    last_row = filled.shape[0] - 1
    last_col = filled.shape[1] - 1
    for row_index in range(filled.shape[0]):
        if background[row_index, 0]:
            stack.append((row_index, 0))
        if background[row_index, last_col]:
            stack.append((row_index, last_col))
    for col_index in range(filled.shape[1]):
        if background[0, col_index]:
            stack.append((0, col_index))
        if background[last_row, col_index]:
            stack.append((last_row, col_index))

    while stack:
        row_index, col_index = stack.pop()
        if (
            row_index < 0
            or row_index > last_row
            or col_index < 0
            or col_index > last_col
            or visited[row_index, col_index]
            or not background[row_index, col_index]
        ):
            continue
        visited[row_index, col_index] = True
        stack.extend(
            [
                (row_index - 1, col_index),
                (row_index + 1, col_index),
                (row_index, col_index - 1),
                (row_index, col_index + 1),
            ]
        )

    holes = background & ~visited
    filled[holes] = True
    return filled


def compute_isodose_volume_within_structure_margin_cc(
    ct: CTVolume,
    dose: DoseVolume,
    structure: StructureSliceContours,
    threshold_gy: float,
    *,
    proximity_mm: float = 5.0,
    structure_mask_cache: Optional[Dict[int, np.ndarray]] = None,
    options: DVHCalculationOptions = HIGH_ACCURACY_DVH_OPTIONS,
) -> float:
    if _should_use_srs_intensive_options(ct, dose, structure, structure_mask_cache):
        model = _build_srs_interpolated_occupancy_model(
            ct,
            dose,
            structure,
            structure_mask_cache=structure_mask_cache,
            options=SRS_INTENSIVE_DVH_OPTIONS,
        )
        if model is None or threshold_gy <= 0.0:
            return 0.0

        occupancy_stack = np.asarray(model["occupancy_stack"], dtype=np.float32)
        if not np.any(occupancy_stack > 1e-6):
            return 0.0

        z_samples_mm = np.asarray(model["z_samples_mm"], dtype=np.float32)
        z_step_mm = float(model["z_step_mm"])
        row_coords_rc = np.asarray(model["row_coords_rc"], dtype=np.float32)
        col_coords_rc = np.asarray(model["col_coords_rc"], dtype=np.float32)
        cell_area_mm2 = float(model["cell_area_mm2"])
        row_spacing_mm = float(model["row_spacing_mm"])
        col_spacing_mm = float(model["col_spacing_mm"])

        ptv_mask = occupancy_stack > 1e-6
        dose_stack = np.zeros_like(occupancy_stack, dtype=np.float32)
        dose_context = _get_dose_sampling_context(dose)
        for z_index, z_mm in enumerate(z_samples_mm):
            dose_stack[z_index] = _sample_dose_plane_virtual_rc(
                dose,
                row_coords_rc.reshape(-1),
                col_coords_rc.reshape(-1),
                float(z_mm),
                linear_interpolation=options.use_linear_dose_interpolation,
                dose_context=dose_context,
            ).reshape(row_coords_rc.shape)
        binary_isodose_mask = dose_stack >= float(threshold_gy)

        try:
            from scipy.ndimage import binary_fill_holes, distance_transform_edt  # type: ignore
        except Exception:  # pragma: no cover - optional runtime dependency
            binary_fill_holes = None
            distance_transform_edt = None

        if binary_fill_holes is not None:
            filled_isodose_mask = np.asarray(binary_fill_holes(binary_isodose_mask), dtype=bool)
        else:
            filled_isodose_mask = np.asarray(binary_isodose_mask, dtype=bool)
            for z_index in range(filled_isodose_mask.shape[0]):
                if np.any(filled_isodose_mask[z_index]):
                    filled_isodose_mask[z_index] = fill_binary_holes_2d(filled_isodose_mask[z_index])

        gradient_z, gradient_row, gradient_col = np.gradient(
            dose_stack.astype(np.float32, copy=False),
            z_step_mm,
            row_spacing_mm,
            col_spacing_mm,
            edge_order=1,
        )
        gradient_magnitude = np.sqrt(
            gradient_z * gradient_z
            + gradient_row * gradient_row
            + gradient_col * gradient_col
        )
        voxel_diagonal_mm = float(np.sqrt(z_step_mm * z_step_mm + row_spacing_mm * row_spacing_mm + col_spacing_mm * col_spacing_mm))
        signed_distance_mm = (dose_stack - float(threshold_gy)) / np.maximum(gradient_magnitude, 1e-3)
        fractional_isodose = np.clip(
            0.5 + signed_distance_mm / max(voxel_diagonal_mm, 1e-6),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        fractional_isodose = np.maximum(
            fractional_isodose,
            filled_isodose_mask.astype(np.float32, copy=False),
        )

        if distance_transform_edt is not None:
            proximity_mask = distance_transform_edt(
                ~ptv_mask,
                sampling=(z_step_mm, row_spacing_mm, col_spacing_mm),
            ) <= proximity_mm
        else:
            proximity_mask = np.zeros_like(ptv_mask, dtype=bool)
            z_radius = int(np.ceil(proximity_mm / max(z_step_mm, 1e-6)))
            row_radius = int(np.ceil(proximity_mm / max(row_spacing_mm, 1e-6)))
            col_radius = int(np.ceil(proximity_mm / max(col_spacing_mm, 1e-6)))
            z_indices, row_indices, col_indices = np.where(ptv_mask)
            for z_index, row_index, col_index in zip(z_indices, row_indices, col_indices):
                proximity_mask[
                    max(0, z_index - z_radius): min(ptv_mask.shape[0], z_index + z_radius + 1),
                    max(0, row_index - row_radius): min(ptv_mask.shape[1], row_index + row_radius + 1),
                    max(0, col_index - col_radius): min(ptv_mask.shape[2], col_index + col_radius + 1),
                ] = True

        limited_fractional_isodose = fractional_isodose * proximity_mask.astype(np.float32, copy=False)
        return (
            float(np.sum(limited_fractional_isodose))
            * cell_area_mm2
            * z_step_mm
            / 1000.0
        )

    if threshold_gy <= 0.0 or not structure.points_rc_by_slice:
        return 0.0

    metrics = estimate_structure_geometry_metrics(
        ct,
        dose,
        structure,
        structure_mask_cache=structure_mask_cache,
    )
    decision = (
        compute_oversampling_factor_from_metrics(metrics.rss, metrics.complexity, options)
        if options.automatic_oversampling
        else None
    )
    scale = (
        float(decision.oversampling_factor)
        if decision is not None
        else float(options.fixed_oversampling_factor)
    )
    scale = float(np.clip(scale, options.minimum_oversampling_factor, options.maximum_oversampling_factor))

    ct_context = _get_ct_geometry_context(ct)
    dose_context = _get_dose_sampling_context(dose)
    slice_thicknesses = _slice_thicknesses_mm(ct.z_positions_mm, float(ct.spacing_xyz_mm[2]))

    contour_payloads: List[Tuple[int, float, Sequence[np.ndarray], int, int, np.ndarray]] = []
    global_row_min: Optional[int] = None
    global_row_max: Optional[int] = None
    global_col_min: Optional[int] = None
    global_col_max: Optional[int] = None

    for slice_index in sorted(structure.points_rc_by_slice):
        contours_rc = structure.points_rc_by_slice.get(slice_index, [])
        if not contours_rc:
            continue
        slice_transform = _build_ct_slice_to_dose_transform(ct_context, dose_context, slice_index)
        contours_dose_rc = [
            (
                np.asarray(contour, dtype=np.float32) @ slice_transform.transform_rc_to_dose_rc.T
                + slice_transform.offset_rc_to_dose_rc[None, :]
            ).astype(np.float32, copy=False)
            for contour in contours_rc
        ]
        occupancy, row_offset, col_offset = _build_local_occupancy_grid(
            contours_dose_rc,
            scale,
            use_fractional_labelmap=options.use_fractional_labelmap,
            fractional_subdivisions=options.fractional_subdivisions,
            max_border_batch_points=options.max_border_batch_points,
        )
        if occupancy.size == 0 or not np.any(occupancy > 0.0):
            continue
        row_min = int(row_offset)
        row_max = int(row_offset + occupancy.shape[0] - 1)
        col_min = int(col_offset)
        col_max = int(col_offset + occupancy.shape[1] - 1)
        global_row_min = row_min if global_row_min is None else min(global_row_min, row_min)
        global_row_max = row_max if global_row_max is None else max(global_row_max, row_max)
        global_col_min = col_min if global_col_min is None else min(global_col_min, col_min)
        global_col_max = col_max if global_col_max is None else max(global_col_max, col_max)
        contour_payloads.append(
            (slice_index, slice_transform.plane_position_mm, contours_dose_rc, row_offset, col_offset, occupancy)
        )

    if not contour_payloads or global_row_min is None or global_col_min is None:
        return 0.0

    in_plane_spacing_row_mm = float(dose_context.spacing_row_mm / scale)
    in_plane_spacing_col_mm = float(dose_context.spacing_col_mm / scale)
    margin_rows = int(np.ceil(proximity_mm / max(in_plane_spacing_row_mm, 1e-6)))
    margin_cols = int(np.ceil(proximity_mm / max(in_plane_spacing_col_mm, 1e-6)))
    margin_z = int(np.ceil(proximity_mm / max(float(ct.spacing_xyz_mm[2]), 1e-6)))

    slice_indices = sorted({payload[0] for payload in contour_payloads})
    z_start = max(0, slice_indices[0] - margin_z)
    z_end = min(ct.volume_hu.shape[0] - 1, slice_indices[-1] + margin_z)
    row_start = max(0, global_row_min - margin_rows)
    row_end = global_row_max + margin_rows
    col_start = max(0, global_col_min - margin_cols)
    col_end = global_col_max + margin_cols

    local_shape = (
        z_end - z_start + 1,
        row_end - row_start + 1,
        col_end - col_start + 1,
    )
    ptv_mask_local = np.zeros(local_shape, dtype=bool)
    isodose_mask_local = np.zeros(local_shape, dtype=bool)
    plane_position_by_slice: Dict[int, float] = {}

    payload_by_slice = {payload[0]: payload for payload in contour_payloads}
    flat_rows = np.arange(row_start, row_end + 1, dtype=np.float32)
    flat_cols = np.arange(col_start, col_end + 1, dtype=np.float32)
    row_coords_grid, col_coords_grid = np.meshgrid(flat_rows, flat_cols, indexing="ij")
    row_coords_flat = ((row_coords_grid + 0.5) / scale).reshape(-1).astype(np.float32, copy=False)
    col_coords_flat = ((col_coords_grid + 0.5) / scale).reshape(-1).astype(np.float32, copy=False)

    for slice_index in range(z_start, z_end + 1):
        local_slice_index = slice_index - z_start
        payload = payload_by_slice.get(slice_index)
        if payload is not None:
            _, plane_position_mm, _contours_dose_rc, row_offset, col_offset, occupancy = payload
            occupancy_mask = occupancy > 0.0
            rr0 = int(row_offset - row_start)
            cc0 = int(col_offset - col_start)
            rr1 = rr0 + occupancy.shape[0]
            cc1 = cc0 + occupancy.shape[1]
            ptv_mask_local[local_slice_index, rr0:rr1, cc0:cc1] |= occupancy_mask
        else:
            plane_position_mm = plane_position_by_slice.get(slice_index)
            if plane_position_mm is None:
                slice_transform = _build_ct_slice_to_dose_transform(ct_context, dose_context, slice_index)
                plane_position_mm = slice_transform.plane_position_mm
        plane_position_by_slice[slice_index] = plane_position_mm

        dose_values = _sample_dose_plane_virtual_rc(
            dose,
            row_coords_flat,
            col_coords_flat,
            plane_position_mm,
            linear_interpolation=options.use_linear_dose_interpolation,
            dose_context=dose_context,
        ).reshape(local_shape[1], local_shape[2])
        isodose_mask_local[local_slice_index] = dose_values >= float(threshold_gy)

    if not np.any(ptv_mask_local):
        return 0.0

    for slice_offset in range(isodose_mask_local.shape[0]):
        if np.any(isodose_mask_local[slice_offset]):
            isodose_mask_local[slice_offset] = fill_binary_holes_2d(isodose_mask_local[slice_offset])

    proximity_sampling = (
        float(ct.spacing_xyz_mm[2]),
        in_plane_spacing_row_mm,
        in_plane_spacing_col_mm,
    )
    try:
        from scipy.ndimage import distance_transform_edt  # type: ignore
    except Exception:  # pragma: no cover - optional runtime dependency
        distance_transform_edt = None

    if distance_transform_edt is not None:
        proximity_mask = distance_transform_edt(~ptv_mask_local, sampling=proximity_sampling) <= proximity_mm
    else:
        proximity_mask = np.zeros_like(ptv_mask_local, dtype=bool)
        z_radius = int(np.ceil(proximity_mm / max(proximity_sampling[0], 1e-6)))
        for delta_z in range(-z_radius, z_radius + 1):
            dz_mm = delta_z * proximity_sampling[0]
            remaining_xy_mm = max(proximity_mm * proximity_mm - dz_mm * dz_mm, 0.0)
            if remaining_xy_mm < 0.0:
                continue
            row_radius = int(np.ceil(np.sqrt(remaining_xy_mm) / max(in_plane_spacing_row_mm, 1e-6)))
            col_radius = int(np.ceil(np.sqrt(remaining_xy_mm) / max(in_plane_spacing_col_mm, 1e-6)))
            src_z_start = max(0, -delta_z)
            src_z_end = ptv_mask_local.shape[0] - max(0, delta_z)
            dst_z_start = max(0, delta_z)
            dst_z_end = ptv_mask_local.shape[0] - max(0, -delta_z)
            if src_z_start >= src_z_end:
                continue
            for delta_r in range(-row_radius, row_radius + 1):
                for delta_c in range(-col_radius, col_radius + 1):
                    distance_mm = np.sqrt(
                        dz_mm * dz_mm
                        + (delta_r * in_plane_spacing_row_mm) ** 2
                        + (delta_c * in_plane_spacing_col_mm) ** 2
                    )
                    if distance_mm > proximity_mm + 1e-6:
                        continue
                    src_r_start = max(0, -delta_r)
                    src_r_end = ptv_mask_local.shape[1] - max(0, delta_r)
                    dst_r_start = max(0, delta_r)
                    dst_r_end = ptv_mask_local.shape[1] - max(0, -delta_r)
                    src_c_start = max(0, -delta_c)
                    src_c_end = ptv_mask_local.shape[2] - max(0, delta_c)
                    dst_c_start = max(0, delta_c)
                    dst_c_end = ptv_mask_local.shape[2] - max(0, -delta_c)
                    if src_r_start >= src_r_end or src_c_start >= src_c_end:
                        continue
                    proximity_mask[dst_z_start:dst_z_end, dst_r_start:dst_r_end, dst_c_start:dst_c_end] |= (
                        ptv_mask_local[src_z_start:src_z_end, src_r_start:src_r_end, src_c_start:src_c_end]
                    )

    limited_isodose_mask = isodose_mask_local & proximity_mask
    if not np.any(limited_isodose_mask):
        return 0.0

    cell_area_mm2 = in_plane_spacing_row_mm * in_plane_spacing_col_mm
    total_volume_cc = 0.0
    for local_slice_index, slice_index in enumerate(range(z_start, z_end + 1)):
        slice_thickness_mm = (
            float(slice_thicknesses[slice_index])
            if slice_index < slice_thicknesses.size
            else float(ct.spacing_xyz_mm[2])
        )
        total_volume_cc += (
            float(np.count_nonzero(limited_isodose_mask[local_slice_index]))
            * cell_area_mm2
            * slice_thickness_mm
            / 1000.0
        )
    return total_volume_cc


def _compute_fast_dvh_curves(
    ct: CTVolume,
    dose: DoseVolume,
    rtstruct: RTStructData,
    dose_ct_volume: Optional[np.ndarray] = None,
    mask_cache: Optional[List[Dict[int, np.ndarray]]] = None,
) -> List[DVHCurve]:
    if not rtstruct.structures:
        return []

    global_max_dose = float(np.nanmax(dose.dose_gy))
    voxel_volume_cc = float(np.prod(ct.spacing_xyz_mm) / 1000.0)

    use_cached_dose = (
        dose_ct_volume is not None
        and dose_ct_volume.shape == ct.volume_hu.shape
    )
    use_cached_masks = (
        mask_cache is not None
        and len(mask_cache) == len(rtstruct.structures)
    )

    dose_samples_by_structure: List[List[np.ndarray]] = [[] for _ in rtstruct.structures]
    relevant_slices = sorted(
        {
            slice_index
            for structure in rtstruct.structures
            for slice_index in structure.points_rc_by_slice.keys()
        }
    )

    for slice_index in relevant_slices:
        if use_cached_dose:
            dose_plane = dose_ct_volume[slice_index]
        else:
            dose_plane = sample_dose_to_ct_slice(ct, dose, slice_index)
        for struct_idx, structure in enumerate(rtstruct.structures):
            if slice_index not in structure.points_rc_by_slice:
                continue
            if use_cached_masks:
                mask = mask_cache[struct_idx].get(slice_index)
                if mask is None:
                    mask = build_structure_slice_mask(structure, slice_index, ct.rows, ct.cols)
            else:
                mask = build_structure_slice_mask(structure, slice_index, ct.rows, ct.cols)
            if np.any(mask):
                dose_samples_by_structure[struct_idx].append(dose_plane[mask])

    curves: List[DVHCurve] = []
    for structure, dose_samples in zip(rtstruct.structures, dose_samples_by_structure):
        if not dose_samples:
            continue
        dose_values = np.concatenate(dose_samples).astype(np.float32, copy=False)
        if dose_values.size == 0:
            continue
        dose_bins_gy, volume_pct = compute_cumulative_dvh(dose_values, global_max_dose)
        curves.append(
            DVHCurve(
                name=structure.name,
                color_rgb=structure.color_rgb,
                dose_bins_gy=dose_bins_gy,
                volume_pct=volume_pct,
                voxel_count=int(dose_values.size),
                volume_cc=float(dose_values.size * voxel_volume_cc),
                mean_dose_gy=float(np.mean(dose_values)),
                max_dose_gy=float(np.max(dose_values)),
                min_dose_gy=float(np.min(dose_values)),
            )
        )

    return curves


def compute_dvh_curves(
    ct: CTVolume,
    dose: DoseVolume,
    rtstruct: RTStructData,
    dose_ct_volume: Optional[np.ndarray] = None,
    mask_cache: Optional[List[Dict[int, np.ndarray]]] = None,
    mode: str = "high_accuracy",
) -> List[DVHCurve]:
    if mode == "high_accuracy":
        body_indices = [
            index
            for index, structure in enumerate(rtstruct.structures)
            if normalize_structure_name(structure.name) == "BODY"
        ]
        non_body_indices = [
            index
            for index in range(len(rtstruct.structures))
            if index not in body_indices
        ]

        standard_indices: List[int] = []
        srs_intensive_indices: List[int] = []
        for index in non_body_indices:
            structure = rtstruct.structures[index]
            structure_masks = mask_cache[index] if mask_cache is not None and index < len(mask_cache) else None
            if _should_use_srs_intensive_options(ct, dose, structure, structure_masks):
                srs_intensive_indices.append(index)
            else:
                standard_indices.append(index)

        body_rtstruct = RTStructData(
            structures=[rtstruct.structures[index] for index in body_indices],
            frame_of_reference_uid=rtstruct.frame_of_reference_uid,
        )
        standard_rtstruct = RTStructData(
            structures=[rtstruct.structures[index] for index in standard_indices],
            frame_of_reference_uid=rtstruct.frame_of_reference_uid,
        )
        srs_intensive_rtstruct = RTStructData(
            structures=[rtstruct.structures[index] for index in srs_intensive_indices],
            frame_of_reference_uid=rtstruct.frame_of_reference_uid,
        )

        fast_body_curves = (
            _compute_fast_dvh_curves(
                ct,
                dose,
                body_rtstruct,
                dose_ct_volume=dose_ct_volume,
                mask_cache=_subset_mask_cache(mask_cache, body_indices),
            )
            if body_indices
            else []
        )
        high_accuracy_curves = (
            compute_dvh_curves_high_accuracy_module(
                ct,
                dose,
                standard_rtstruct,
                options=HIGH_ACCURACY_DVH_OPTIONS,
                mask_cache=_subset_mask_cache(mask_cache, standard_indices),
            )
            if standard_indices
            else []
        )
        srs_intensive_curves = (
            [
                curve
                for curve in (
                    _compute_srs_interpolated_curve(
                        ct,
                        dose,
                        structure,
                        structure_mask_cache=(
                            mask_cache[index]
                            if mask_cache is not None and index < len(mask_cache)
                            else None
                        ),
                        options=SRS_INTENSIVE_DVH_OPTIONS,
                    )
                    for index, structure in zip(srs_intensive_indices, srs_intensive_rtstruct.structures)
                )
                if curve is not None
            ]
            if srs_intensive_indices
            else []
        )

        curves_by_name = {
            normalize_structure_name(curve.name): curve
            for curve in [*fast_body_curves, *high_accuracy_curves, *srs_intensive_curves]
        }
        return [
            curves_by_name[normalize_structure_name(structure.name)]
            for structure in rtstruct.structures
            if normalize_structure_name(structure.name) in curves_by_name
        ]

    return _compute_fast_dvh_curves(
        ct,
        dose,
        rtstruct,
        dose_ct_volume=dose_ct_volume,
        mask_cache=mask_cache,
    )


def _callable_signature_hash(func) -> str:
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = repr(func)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _options_signature(options: DVHCalculationOptions) -> Dict[str, object]:
    return {
        "automatic_oversampling": bool(options.automatic_oversampling),
        "fixed_oversampling_factor": float(options.fixed_oversampling_factor),
        "use_fractional_labelmap": bool(options.use_fractional_labelmap),
        "fractional_subdivisions": int(options.fractional_subdivisions),
        "use_linear_dose_interpolation": bool(options.use_linear_dose_interpolation),
        "dose_bin_width_gy": float(options.dose_bin_width_gy),
        "max_dose_gy": None if options.max_dose_gy is None else float(options.max_dose_gy),
        "minimum_oversampling_factor": float(options.minimum_oversampling_factor),
        "maximum_oversampling_factor": float(options.maximum_oversampling_factor),
        "max_border_batch_points": int(options.max_border_batch_points),
    }


def get_dvh_method_signature() -> Dict[str, object]:
    return {
        "high_accuracy_options": _options_signature(HIGH_ACCURACY_DVH_OPTIONS),
        "srs_intensive_options": _options_signature(SRS_INTENSIVE_DVH_OPTIONS),
        "srs_small_volume_threshold_cc": float(SRS_SMALL_VOLUME_THRESHOLD_CC),
        "compute_dvh_curves": _callable_signature_hash(compute_dvh_curves),
        "compute_single_structure_high_accuracy_curve": _callable_signature_hash(
            compute_single_structure_high_accuracy_curve
        ),
        "compute_isodose_volume_within_structure_margin_cc": _callable_signature_hash(
            compute_isodose_volume_within_structure_margin_cc
        ),
        "_build_srs_interpolated_occupancy_model": _callable_signature_hash(
            _build_srs_interpolated_occupancy_model
        ),
        "_compute_srs_interpolated_curve": _callable_signature_hash(_compute_srs_interpolated_curve),
    }
