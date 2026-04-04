from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from peer_models import CTVolume, DVHCurve, DoseVolume, RTStructData, StructureSliceContours


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class DVHCalculationOptions:
    """Configuration for DVH generation.

    The implementation borrows three ideas from the supplied references:
    1. Structure-specific oversampling selected by fuzzy logic (Pinter 2020).
    2. Fractional border occupancy instead of strict binary inclusion (Sunderland 2017).
    3. Linear interpolation for D/V metrics from the cumulative DVH curve
       (mirroring the SlicerRT logic).

    Notes
    -----
    The peer review app stores RTSTRUCT contours as slice-aligned planar contours on the
    CT geometry. Because there is no intermediate triangulated surface mesh available in
    the current app, out-of-plane interpolation is approximated as piecewise constant
    within the slab thickness of each contoured CT slice. In-plane oversampling and
    fractional border occupancy are implemented directly on a dose-referenced virtual
    grid, which yields a substantial accuracy improvement over the original whole-voxel
    CT-grid approach while remaining self-contained and dependency-light.
    """

    automatic_oversampling: bool = True
    fixed_oversampling_factor: float = 2.0
    use_fractional_labelmap: bool = True
    fractional_subdivisions: int = 6
    use_linear_dose_interpolation: bool = True
    dose_bin_width_gy: float = 0.2
    max_dose_gy: Optional[float] = None
    minimum_oversampling_factor: float = 0.5
    maximum_oversampling_factor: float = 4.0
    max_border_batch_points: int = 200_000


DEFAULT_DVH_OPTIONS = DVHCalculationOptions()


# -----------------------------------------------------------------------------
# Small geometry utilities
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class _DoseSamplingContext:
    z_positions_mm: np.ndarray
    dose_volume: np.ndarray
    dose_row_cos: np.ndarray
    dose_col_cos: np.ndarray
    dose_normal: np.ndarray
    first_origin: np.ndarray
    first_pos_mm: float
    spacing_row_mm: float
    spacing_col_mm: float


@dataclass(slots=True)
class _CTGeometryContext:
    row_cos: np.ndarray
    col_cos: np.ndarray
    normal: np.ndarray
    slice_origins_xyz_mm: np.ndarray
    spacing_row_mm: float
    spacing_col_mm: float


@dataclass(slots=True)
class _SliceDoseTransform:
    plane_position_mm: float
    transform_rc_to_dose_rc: np.ndarray
    offset_rc_to_dose_rc: np.ndarray


def _get_runtime_cache(obj: object, attr_name: str) -> Dict[object, object]:
    cache = getattr(obj, attr_name, None)
    if isinstance(cache, dict):
        return cache
    cache = {}
    setattr(obj, attr_name, cache)
    return cache


def _get_iop_normalized(iop: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_cos = np.asarray(iop[:3], dtype=np.float64)
    col_cos = np.asarray(iop[3:], dtype=np.float64)
    normal = np.cross(row_cos, col_cos)

    row_norm = np.linalg.norm(row_cos)
    col_norm = np.linalg.norm(col_cos)
    normal_norm = np.linalg.norm(normal)
    if row_norm == 0.0 or col_norm == 0.0 or normal_norm == 0.0:
        raise ValueError("Invalid ImageOrientationPatient vectors.")

    row_cos /= row_norm
    col_cos /= col_norm
    normal /= normal_norm
    return row_cos, col_cos, normal


def _get_ct_geometry_context(ct: CTVolume) -> _CTGeometryContext:
    cached = getattr(ct, "_peer_dvh_geometry_context", None)
    if isinstance(cached, _CTGeometryContext):
        return cached

    row_cos, col_cos, normal = _get_iop_normalized(ct.image_orientation_patient)
    context = _CTGeometryContext(
        row_cos=row_cos.astype(np.float32),
        col_cos=col_cos.astype(np.float32),
        normal=normal.astype(np.float32),
        slice_origins_xyz_mm=np.asarray(ct.slice_origins_xyz_mm, dtype=np.float32),
        spacing_row_mm=float(ct.spacing_xyz_mm[1]),
        spacing_col_mm=float(ct.spacing_xyz_mm[0]),
    )
    setattr(ct, "_peer_dvh_geometry_context", context)
    return context


def _get_dose_sampling_context(dose: DoseVolume) -> _DoseSamplingContext:
    cached = getattr(dose, "_peer_dvh_sampling_context", None)
    if isinstance(cached, _DoseSamplingContext):
        return cached

    dose_volume = np.asarray(dose.dose_gy, dtype=np.float32)
    z_positions_mm, dose_volume = _prepare_z_axis(np.asarray(dose.z_positions_mm, dtype=np.float32), dose_volume)
    dose_row_cos, dose_col_cos, dose_normal = _get_iop_normalized(dose.image_orientation_patient)
    first_origin = np.asarray(dose.slice_origins_xyz_mm[0], dtype=np.float32)
    context = _DoseSamplingContext(
        z_positions_mm=z_positions_mm.astype(np.float32, copy=False),
        dose_volume=dose_volume.astype(np.float32, copy=False),
        dose_row_cos=dose_row_cos.astype(np.float32),
        dose_col_cos=dose_col_cos.astype(np.float32),
        dose_normal=dose_normal.astype(np.float32),
        first_origin=first_origin,
        first_pos_mm=float(np.dot(first_origin.astype(np.float64), dose_normal)),
        spacing_row_mm=float(dose.spacing_xyz_mm[1]),
        spacing_col_mm=float(dose.spacing_xyz_mm[0]),
    )
    setattr(dose, "_peer_dvh_sampling_context", context)
    return context


def _build_ct_slice_to_dose_transform(
    ct_context: _CTGeometryContext,
    dose_context: _DoseSamplingContext,
    slice_index: int,
) -> _SliceDoseTransform:
    ct_origin = ct_context.slice_origins_xyz_mm[slice_index]
    plane_position_mm = float(np.dot(ct_origin.astype(np.float64), dose_context.dose_normal.astype(np.float64)))
    plane_origin = (
        dose_context.first_origin
        + np.float32(plane_position_mm - dose_context.first_pos_mm) * dose_context.dose_normal
    ).astype(np.float32, copy=False)
    rel_origin = (ct_origin - plane_origin).astype(np.float32, copy=False)

    row_basis_xyz = ct_context.col_cos * np.float32(ct_context.spacing_row_mm)
    col_basis_xyz = ct_context.row_cos * np.float32(ct_context.spacing_col_mm)

    transform = np.array(
        [
            [
                float(np.dot(row_basis_xyz, dose_context.dose_col_cos) / dose_context.spacing_row_mm),
                float(np.dot(col_basis_xyz, dose_context.dose_col_cos) / dose_context.spacing_row_mm),
            ],
            [
                float(np.dot(row_basis_xyz, dose_context.dose_row_cos) / dose_context.spacing_col_mm),
                float(np.dot(col_basis_xyz, dose_context.dose_row_cos) / dose_context.spacing_col_mm),
            ],
        ],
        dtype=np.float32,
    )
    offset = np.array(
        [
            float(np.dot(rel_origin, dose_context.dose_col_cos) / dose_context.spacing_row_mm),
            float(np.dot(rel_origin, dose_context.dose_row_cos) / dose_context.spacing_col_mm),
        ],
        dtype=np.float32,
    )
    return _SliceDoseTransform(
        plane_position_mm=plane_position_mm,
        transform_rc_to_dose_rc=transform,
        offset_rc_to_dose_rc=offset,
    )


def _ct_rc_to_patient_xyz(points_rc: np.ndarray, ct: CTVolume, slice_index: int) -> np.ndarray:
    row_cos, col_cos, _ = _get_iop_normalized(ct.image_orientation_patient)
    origin = np.asarray(ct.slice_origins_xyz_mm[slice_index], dtype=np.float64)
    sx = float(ct.spacing_xyz_mm[0])
    sy = float(ct.spacing_xyz_mm[1])

    rows = np.asarray(points_rc[:, 0], dtype=np.float64)
    cols = np.asarray(points_rc[:, 1], dtype=np.float64)
    return origin[None, :] + cols[:, None] * sx * row_cos[None, :] + rows[:, None] * sy * col_cos[None, :]


def _patient_xyz_to_virtual_dose_rc(
    points_xyz: np.ndarray,
    dose: DoseVolume,
    plane_position_mm: float,
) -> np.ndarray:
    dose_row_cos, dose_col_cos, dose_normal = _get_iop_normalized(dose.image_orientation_patient)
    sx = float(dose.spacing_xyz_mm[0])
    sy = float(dose.spacing_xyz_mm[1])
    first_origin = np.asarray(dose.slice_origins_xyz_mm[0], dtype=np.float64)
    first_pos = float(np.dot(first_origin, dose_normal))
    plane_origin = first_origin + (plane_position_mm - first_pos) * dose_normal
    rel = np.asarray(points_xyz, dtype=np.float64) - plane_origin[None, :]
    cols = rel @ dose_row_cos / sx
    rows = rel @ dose_col_cos / sy
    return np.column_stack([rows, cols])


def _points_patient_xyz_from_virtual_dose_rc(
    dose: DoseVolume,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    plane_position_mm: float,
    z_offsets_mm: Optional[np.ndarray] = None,
) -> np.ndarray:
    dose_row_cos, dose_col_cos, dose_normal = _get_iop_normalized(dose.image_orientation_patient)
    sx = float(dose.spacing_xyz_mm[0])
    sy = float(dose.spacing_xyz_mm[1])
    first_origin = np.asarray(dose.slice_origins_xyz_mm[0], dtype=np.float64)
    first_pos = float(np.dot(first_origin, dose_normal))
    plane_origin = first_origin + (plane_position_mm - first_pos) * dose_normal

    row_coords = np.asarray(row_coords, dtype=np.float64)
    col_coords = np.asarray(col_coords, dtype=np.float64)

    base = (
        plane_origin[None, :]
        + col_coords[:, None] * sx * dose_row_cos[None, :]
        + row_coords[:, None] * sy * dose_col_cos[None, :]
    )
    if z_offsets_mm is None or len(z_offsets_mm) == 0:
        return base

    z_offsets_mm = np.asarray(z_offsets_mm, dtype=np.float64)
    return base[:, None, :] + z_offsets_mm[None, :, None] * dose_normal[None, None, :]


def _slice_thicknesses_mm(z_positions_mm: np.ndarray, fallback_spacing_mm: float) -> np.ndarray:
    z = np.asarray(z_positions_mm, dtype=np.float64)
    if z.size == 0:
        return np.zeros(0, dtype=np.float64)
    if z.size == 1:
        return np.array([abs(float(fallback_spacing_mm))], dtype=np.float64)

    boundaries = np.empty(z.size + 1, dtype=np.float64)
    boundaries[1:-1] = 0.5 * (z[:-1] + z[1:])
    boundaries[0] = z[0] - 0.5 * (z[1] - z[0])
    boundaries[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
    return np.abs(np.diff(boundaries))


def _dose_z_sampling_offsets_mm(thickness_mm: float, oversampling_factor: float) -> np.ndarray:
    z_subdivisions = max(1, int(round(max(1.0, oversampling_factor))))
    if z_subdivisions == 1 or thickness_mm <= 0.0:
        return np.array([0.0], dtype=np.float32)
    return (((np.arange(z_subdivisions, dtype=np.float32) + 0.5) / z_subdivisions) - 0.5) * np.float32(thickness_mm)


# -----------------------------------------------------------------------------
# Mask rasterization / point-in-polygon
# -----------------------------------------------------------------------------


def _rasterize_polygon_mask_local(contour_rc: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if contour_rc.shape[0] < 3 or rows <= 0 or cols <= 0:
        return np.zeros((max(rows, 0), max(cols, 0)), dtype=bool)

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
    mask[r_min : r_max + 1, c_min : c_max + 1] = inside
    return mask


def _build_scaled_local_mask(contours_rc: Sequence[np.ndarray], scale: float) -> Tuple[np.ndarray, int, int]:
    if not contours_rc:
        return np.zeros((0, 0), dtype=bool), 0, 0

    scaled_contours = [np.asarray(contour, dtype=np.float32) * np.float32(scale) for contour in contours_rc if contour.shape[0] >= 3]
    if not scaled_contours:
        return np.zeros((0, 0), dtype=bool), 0, 0

    all_rows = np.concatenate([contour[:, 0] for contour in scaled_contours])
    all_cols = np.concatenate([contour[:, 1] for contour in scaled_contours])
    r_min = int(np.floor(np.min(all_rows)))
    r_max = int(np.ceil(np.max(all_rows)))
    c_min = int(np.floor(np.min(all_cols)))
    c_max = int(np.ceil(np.max(all_cols)))
    rows = max(0, r_max - r_min + 1)
    cols = max(0, c_max - c_min + 1)
    if rows == 0 or cols == 0:
        return np.zeros((0, 0), dtype=bool), 0, 0

    mask = np.zeros((rows, cols), dtype=bool)
    shift = np.array([float(r_min), float(c_min)], dtype=np.float32)
    for contour in scaled_contours:
        mask ^= _rasterize_polygon_mask_local(contour - shift[None, :], rows, cols)
    return mask, r_min, c_min


def _mask_interior_border(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if mask.size == 0:
        return mask.copy(), mask.copy()
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        mask
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
    )
    border = mask & ~interior
    return interior, border


def _points_in_contours_xor(row_coords: np.ndarray, col_coords: np.ndarray, contours_rc: Sequence[np.ndarray]) -> np.ndarray:
    row_coords = np.asarray(row_coords, dtype=np.float32)
    col_coords = np.asarray(col_coords, dtype=np.float32)
    inside = np.zeros(row_coords.shape, dtype=bool)

    for contour in contours_rc:
        contour = np.asarray(contour, dtype=np.float32)
        if contour.shape[0] < 3:
            continue
        rr = contour[:, 0]
        cc = contour[:, 1]
        next_rr = np.roll(rr, -1)
        next_cc = np.roll(cc, -1)
        contour_inside = np.zeros(row_coords.shape, dtype=bool)
        for r0, c0, r1, c1 in zip(rr, cc, next_rr, next_cc):
            if np.isclose(r0, r1):
                continue
            intersects = ((r0 > row_coords) != (r1 > row_coords))
            cross_cols = (c1 - c0) * (row_coords - r0) / (r1 - r0) + c0
            contour_inside ^= intersects & (col_coords < cross_cols)
        inside ^= contour_inside

    return inside


# -----------------------------------------------------------------------------
# Structure metrics for fuzzy oversampling
# -----------------------------------------------------------------------------


def _build_structure_slice_mask(structure: StructureSliceContours, slice_index: int, rows: int, cols: int) -> np.ndarray:
    mask = np.zeros((rows, cols), dtype=bool)
    for contour_rc in structure.points_rc_by_slice.get(slice_index, []):
        mask ^= _rasterize_polygon_mask_local(np.asarray(contour_rc, dtype=np.float64), rows, cols)
    return mask


def _ensure_mask_cache(
    rtstruct: RTStructData,
    rows: int,
    cols: int,
    mask_cache: Optional[List[Dict[int, np.ndarray]]],
) -> List[Dict[int, np.ndarray]]:
    if mask_cache is not None and len(mask_cache) == len(rtstruct.structures):
        return mask_cache

    rebuilt: List[Dict[int, np.ndarray]] = []
    for structure in rtstruct.structures:
        structure_masks: Dict[int, np.ndarray] = {}
        for slice_index in structure.points_rc_by_slice:
            structure_masks[slice_index] = _build_structure_slice_mask(structure, slice_index, rows, cols)
        rebuilt.append(structure_masks)
    return rebuilt


@dataclass(slots=True)
class StructureGeometryMetrics:
    volume_mm3: float
    surface_area_mm2: float
    rss: float
    complexity: float


def _mask_perimeter_mm(mask: np.ndarray, sx_mm: float, sy_mm: float) -> float:
    if mask.size == 0:
        return 0.0
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    vertical_edges = padded[1:-1, 1:] != padded[1:-1, :-1]
    horizontal_edges = padded[1:, 1:-1] != padded[:-1, 1:-1]
    return float(np.count_nonzero(vertical_edges) * sy_mm + np.count_nonzero(horizontal_edges) * sx_mm)


def estimate_structure_geometry_metrics(
    ct: CTVolume,
    dose: DoseVolume,
    structure: StructureSliceContours,
    structure_mask_cache: Optional[Dict[int, np.ndarray]] = None,
) -> StructureGeometryMetrics:
    sx_mm = float(ct.spacing_xyz_mm[0])
    sy_mm = float(ct.spacing_xyz_mm[1])
    slice_thicknesses = _slice_thicknesses_mm(ct.z_positions_mm, float(ct.spacing_xyz_mm[2]))
    pixel_area_mm2 = sx_mm * sy_mm
    dose_voxel_mm3 = float(np.prod(dose.spacing_xyz_mm))

    total_volume_mm3 = 0.0
    lateral_surface_mm2 = 0.0
    first_area_mm2 = 0.0
    last_area_mm2 = 0.0
    first_seen = False

    for slice_index in sorted(structure.points_rc_by_slice):
        mask = None
        if structure_mask_cache is not None:
            mask = structure_mask_cache.get(slice_index)
        if mask is None:
            mask = _build_structure_slice_mask(structure, slice_index, ct.rows, ct.cols)
        if not np.any(mask):
            continue

        area_mm2 = float(np.count_nonzero(mask) * pixel_area_mm2)
        thickness_mm = float(slice_thicknesses[slice_index]) if slice_index < slice_thicknesses.size else float(ct.spacing_xyz_mm[2])
        perimeter_mm = _mask_perimeter_mm(mask, sx_mm, sy_mm)

        total_volume_mm3 += area_mm2 * thickness_mm
        lateral_surface_mm2 += perimeter_mm * thickness_mm

        if not first_seen:
            first_area_mm2 = area_mm2
            first_seen = True
        last_area_mm2 = area_mm2

    cap_surface_mm2 = first_area_mm2 + last_area_mm2 if first_seen else 0.0
    surface_area_mm2 = max(lateral_surface_mm2 + cap_surface_mm2, 1e-6)
    volume_mm3 = max(total_volume_mm3, 0.0)

    if volume_mm3 <= 0.0:
        rss = 0.0
        complexity = 0.0
    else:
        rss = float(np.cbrt(volume_mm3 / max(dose_voxel_mm3, 1e-6)))
        # Sphere-normalized surface-to-volume index (equivalent form of the Alyassin-style NSI).
        nsi = float(surface_area_mm2 / np.cbrt(36.0 * np.pi * volume_mm3 * volume_mm3))
        complexity = max(0.0, nsi - 1.0)

    return StructureGeometryMetrics(
        volume_mm3=volume_mm3,
        surface_area_mm2=surface_area_mm2,
        rss=rss,
        complexity=complexity,
    )


# -----------------------------------------------------------------------------
# Fuzzy oversampling factor (Pinter 2020 style)
# -----------------------------------------------------------------------------


def _trapmf(x: np.ndarray | float, a: float, b: float, c: float, d: float) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(values, dtype=np.float64)
    if b > a:
        rising = (values > a) & (values < b)
        out[rising] = (values[rising] - a) / (b - a)
    plateau = (values >= b) & (values <= c)
    out[plateau] = 1.0
    if d > c:
        falling = (values > c) & (values < d)
        out[falling] = (d - values[falling]) / (d - c)
    out[values <= a] = np.maximum(out[values <= a], 1.0 if a == b else 0.0)
    out[values >= d] = np.maximum(out[values >= d], 1.0 if c == d else 0.0)
    return np.clip(out, 0.0, 1.0)


@dataclass(slots=True)
class OversamplingDecision:
    oversampling_factor: float
    power_of_two: float
    rss: float
    complexity: float
    memberships: Dict[str, float] = field(default_factory=dict)


# The following piecewise memberships are visually reconstructed from Fig. 2 in Pinter 2020.
# They are intentionally smooth and conservative. Exact breakpoint values are not published as a table.
_RSS_VERY_SMALL = (0.0, 0.0, 7.0, 12.0)
_RSS_SMALL = (7.0, 10.0, 12.0, 18.0)
_RSS_MEDIUM = (14.0, 18.0, 34.0, 72.0)
_RSS_LARGE = (36.0, 72.0, 100.0, 100.0)

_COMPLEXITY_LOW = (0.0, 0.0, 0.2, 0.6)
_COMPLEXITY_HIGH = (0.2, 0.6, 2.0, 2.0)

_OUTPUT_LOW = (-1.0, -1.0, -0.75, -0.25)
_OUTPUT_NORMAL = (-0.75, -0.25, 0.25, 0.75)
_OUTPUT_HIGH = (0.25, 0.75, 1.25, 1.75)
_OUTPUT_VERY_HIGH = (1.25, 1.75, 2.0, 2.0)


def compute_oversampling_factor_from_metrics(
    rss: float,
    complexity: float,
    options: DVHCalculationOptions = DEFAULT_DVH_OPTIONS,
) -> OversamplingDecision:
    rss = float(max(0.0, rss))
    complexity = float(max(0.0, complexity))

    rss_vs = float(_trapmf(rss, *_RSS_VERY_SMALL))
    rss_s = float(_trapmf(rss, *_RSS_SMALL))
    rss_m = float(_trapmf(rss, *_RSS_MEDIUM))
    rss_l = float(_trapmf(rss, *_RSS_LARGE))
    comp_low = float(_trapmf(complexity, *_COMPLEXITY_LOW))
    comp_high = float(_trapmf(complexity, *_COMPLEXITY_HIGH))

    # Mamdani rules from Pinter 2020, section 2.5.
    rule_very_high = rss_vs
    rule_high = max(min(rss_s, comp_high), min(rss_m, comp_high))
    rule_normal = max(min(rss_s, comp_low), min(rss_m, comp_low))
    rule_low = rss_l

    x = np.linspace(-1.0, 2.0, 3001, dtype=np.float64)
    aggregated = np.maximum.reduce(
        [
            np.minimum(_trapmf(x, *_OUTPUT_LOW), rule_low),
            np.minimum(_trapmf(x, *_OUTPUT_NORMAL), rule_normal),
            np.minimum(_trapmf(x, *_OUTPUT_HIGH), rule_high),
            np.minimum(_trapmf(x, *_OUTPUT_VERY_HIGH), rule_very_high),
        ]
    )

    if float(np.sum(aggregated)) <= 0.0:
        power = 0.0
    else:
        power = float(np.sum(x * aggregated) / np.sum(aggregated))

    rounded_power = int(np.round(np.clip(power, -1.0, 2.0)))
    factor = float(2.0 ** rounded_power)
    factor = float(np.clip(factor, options.minimum_oversampling_factor, options.maximum_oversampling_factor))

    return OversamplingDecision(
        oversampling_factor=factor,
        power_of_two=power,
        rss=rss,
        complexity=complexity,
        memberships={
            "rss_very_small": rss_vs,
            "rss_small": rss_s,
            "rss_medium": rss_m,
            "rss_large": rss_l,
            "complexity_low": comp_low,
            "complexity_high": comp_high,
            "rule_low": rule_low,
            "rule_normal": rule_normal,
            "rule_high": rule_high,
            "rule_very_high": rule_very_high,
        },
    )


# -----------------------------------------------------------------------------
# Dose sampling
# -----------------------------------------------------------------------------


def _prepare_z_axis(z_positions_mm: np.ndarray, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z_positions_mm = np.asarray(z_positions_mm, dtype=np.float32)
    if z_positions_mm.size <= 1:
        return z_positions_mm.copy(), volume
    if z_positions_mm[0] <= z_positions_mm[-1]:
        return z_positions_mm.copy(), volume
    return z_positions_mm[::-1].copy(), volume[::-1].copy()


def _bilinear_sample_plane(plane: np.ndarray, row_coords: np.ndarray, col_coords: np.ndarray) -> np.ndarray:
    plane = np.asarray(plane, dtype=np.float32)
    row_coords = np.asarray(row_coords, dtype=np.float32)
    col_coords = np.asarray(col_coords, dtype=np.float32)
    out = np.zeros(row_coords.shape, dtype=np.float32)

    valid = (
        (row_coords >= 0.0)
        & (row_coords <= plane.shape[0] - 1)
        & (col_coords >= 0.0)
        & (col_coords <= plane.shape[1] - 1)
    )
    if not np.any(valid):
        return out

    rr = row_coords[valid]
    cc = col_coords[valid]
    r0 = np.floor(rr).astype(np.int64)
    c0 = np.floor(cc).astype(np.int64)
    r1 = np.clip(r0 + 1, 0, plane.shape[0] - 1)
    c1 = np.clip(c0 + 1, 0, plane.shape[1] - 1)
    dr = rr - r0
    dc = cc - c0

    top = (1.0 - dc) * plane[r0, c0] + dc * plane[r0, c1]
    bottom = (1.0 - dc) * plane[r1, c0] + dc * plane[r1, c1]
    out[valid] = ((1.0 - dr) * top + dr * bottom).astype(np.float32, copy=False)
    return out


def _nearest_sample_plane(plane: np.ndarray, row_coords: np.ndarray, col_coords: np.ndarray) -> np.ndarray:
    plane = np.asarray(plane, dtype=np.float32)
    row_coords = np.asarray(row_coords, dtype=np.float32)
    col_coords = np.asarray(col_coords, dtype=np.float32)
    out = np.zeros(row_coords.shape, dtype=np.float32)

    rr = np.rint(row_coords).astype(np.int64)
    cc = np.rint(col_coords).astype(np.int64)
    valid = (
        (rr >= 0)
        & (rr < plane.shape[0])
        & (cc >= 0)
        & (cc < plane.shape[1])
    )
    if np.any(valid):
        out[valid] = plane[rr[valid], cc[valid]]
    return out


def _sample_dose_plane_virtual_rc(
    dose: DoseVolume,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    plane_position_mm: float,
    *,
    linear_interpolation: bool = True,
    dose_context: Optional[_DoseSamplingContext] = None,
) -> np.ndarray:
    row_coords = np.asarray(row_coords, dtype=np.float32)
    col_coords = np.asarray(col_coords, dtype=np.float32)
    if row_coords.size == 0 or col_coords.size == 0:
        return np.zeros(row_coords.shape, dtype=np.float32)

    context = dose_context if dose_context is not None else _get_dose_sampling_context(dose)
    z_positions_mm = context.z_positions_mm
    dose_volume = context.dose_volume
    if z_positions_mm.size == 0:
        return np.zeros(row_coords.shape, dtype=np.float32)

    if z_positions_mm.size == 1:
        sampler = _bilinear_sample_plane if linear_interpolation else _nearest_sample_plane
        return sampler(dose_volume[0], row_coords, col_coords)

    continuous_k = float(
        np.interp(
            float(plane_position_mm),
            z_positions_mm,
            np.arange(z_positions_mm.size, dtype=np.float32),
            left=-1.0,
            right=float(z_positions_mm.size),
        )
    )
    if continuous_k < 0.0 or continuous_k > float(z_positions_mm.size - 1):
        return np.zeros(row_coords.shape, dtype=np.float32)

    if linear_interpolation:
        if np.isclose(continuous_k, 0.0):
            return _bilinear_sample_plane(dose_volume[0], row_coords, col_coords)
        if np.isclose(continuous_k, float(z_positions_mm.size - 1)):
            return _bilinear_sample_plane(dose_volume[-1], row_coords, col_coords)

        k0 = int(np.floor(continuous_k))
        k1 = k0 + 1
        if k0 < 0 or k1 >= z_positions_mm.size:
            return np.zeros(row_coords.shape, dtype=np.float32)
        frac = continuous_k - float(k0)
        sampled0 = _bilinear_sample_plane(dose_volume[k0], row_coords, col_coords)
        sampled1 = _bilinear_sample_plane(dose_volume[k1], row_coords, col_coords)
        return ((1.0 - frac) * sampled0 + frac * sampled1).astype(np.float32, copy=False)

    k = int(np.clip(np.rint(continuous_k), 0, z_positions_mm.size - 1))
    return _nearest_sample_plane(dose_volume[k], row_coords, col_coords)


def trilinear_sample_dose_patient_xyz(dose: DoseVolume, points_xyz: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.size == 0:
        return np.zeros(0, dtype=np.float64)

    dose_row_cos, dose_col_cos, dose_normal = _get_iop_normalized(dose.image_orientation_patient)
    sx = float(dose.spacing_xyz_mm[0])
    sy = float(dose.spacing_xyz_mm[1])
    first_origin = np.asarray(dose.slice_origins_xyz_mm[0], dtype=np.float64)
    first_pos = float(np.dot(first_origin, dose_normal))

    z_positions_mm, dose_volume = _prepare_z_axis(dose.z_positions_mm, dose.dose_gy)
    if z_positions_mm.size == 0:
        return np.zeros(points_xyz.shape[0], dtype=np.float64)

    plane_positions_mm = points_xyz @ dose_normal
    # Continuous slice coordinate by 1-D interpolation along the dose normal.
    continuous_k = np.interp(
        plane_positions_mm,
        z_positions_mm,
        np.arange(z_positions_mm.size, dtype=np.float64),
        left=-1.0,
        right=float(z_positions_mm.size),
    )

    plane_origins = first_origin[None, :] + (plane_positions_mm - first_pos)[:, None] * dose_normal[None, :]
    rel = points_xyz - plane_origins
    col_coords = rel @ dose_row_cos / sx
    row_coords = rel @ dose_col_cos / sy

    if z_positions_mm.size == 1:
        return _bilinear_sample_plane(dose_volume[0], row_coords, col_coords)

    out = np.zeros(points_xyz.shape[0], dtype=np.float64)

    exact_first = np.isclose(continuous_k, 0.0)
    if np.any(exact_first):
        out[exact_first] = _bilinear_sample_plane(dose_volume[0], row_coords[exact_first], col_coords[exact_first])

    exact_last = np.isclose(continuous_k, z_positions_mm.size - 1.0)
    if np.any(exact_last):
        out[exact_last] = _bilinear_sample_plane(dose_volume[-1], row_coords[exact_last], col_coords[exact_last])

    k0 = np.floor(continuous_k).astype(np.int64)
    k1 = k0 + 1
    valid = (~exact_first) & (~exact_last) & (k0 >= 0) & (k1 < z_positions_mm.size)
    if not np.any(valid):
        return out

    rr = row_coords[valid]
    cc = col_coords[valid]
    p0 = dose_volume[k0[valid]]
    p1 = dose_volume[k1[valid]]
    frac = continuous_k[valid] - k0[valid]

    inside = (
        (rr >= 0.0)
        & (rr <= p0.shape[1] - 1)
        & (cc >= 0.0)
        & (cc <= p0.shape[2] - 1)
    )
    if not np.any(inside):
        return out

    rr = rr[inside]
    cc = cc[inside]
    p0 = p0[inside]
    p1 = p1[inside]
    frac = frac[inside]
    valid_indices = np.where(valid)[0][inside]

    r0 = np.floor(rr).astype(np.int64)
    c0 = np.floor(cc).astype(np.int64)
    r1 = np.clip(r0 + 1, 0, p0.shape[1] - 1)
    c1 = np.clip(c0 + 1, 0, p0.shape[2] - 1)
    dr = rr - r0
    dc = cc - c0
    plane_index = np.arange(rr.size, dtype=np.int64)

    top0 = (1.0 - dc) * p0[plane_index, r0, c0] + dc * p0[plane_index, r0, c1]
    bottom0 = (1.0 - dc) * p0[plane_index, r1, c0] + dc * p0[plane_index, r1, c1]
    sampled0 = (1.0 - dr) * top0 + dr * bottom0

    top1 = (1.0 - dc) * p1[plane_index, r0, c0] + dc * p1[plane_index, r0, c1]
    bottom1 = (1.0 - dc) * p1[plane_index, r1, c0] + dc * p1[plane_index, r1, c1]
    sampled1 = (1.0 - dr) * top1 + dr * bottom1

    out[valid_indices] = (1.0 - frac) * sampled0 + frac * sampled1
    return out


# -----------------------------------------------------------------------------
# Weighted DVH accumulation
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class _WeightedHistogramAccumulator:
    bin_edges_gy: np.ndarray
    hist_cc: np.ndarray
    total_volume_cc: float = 0.0
    weighted_dose_sum_gy_cc: float = 0.0
    min_dose_gy: float = np.inf
    max_dose_gy: float = -np.inf
    sample_count: int = 0

    def add(self, dose_values_gy: np.ndarray, weights_cc: np.ndarray) -> None:
        dose_values_gy = np.asarray(dose_values_gy, dtype=np.float64)
        weights_cc = np.asarray(weights_cc, dtype=np.float64)
        if dose_values_gy.size == 0 or weights_cc.size == 0:
            return

        positive = weights_cc > 0.0
        if not np.any(positive):
            return
        dose_values_gy = dose_values_gy[positive]
        weights_cc = weights_cc[positive]

        max_edge = float(self.bin_edges_gy[-1])
        clipped = np.clip(dose_values_gy, 0.0, max_edge)
        self.hist_cc += np.histogram(clipped, bins=self.bin_edges_gy, weights=weights_cc)[0]
        self.total_volume_cc += float(np.sum(weights_cc))
        self.weighted_dose_sum_gy_cc += float(np.dot(dose_values_gy, weights_cc))
        self.min_dose_gy = min(self.min_dose_gy, float(np.min(dose_values_gy)))
        self.max_dose_gy = max(self.max_dose_gy, float(np.max(dose_values_gy)))
        self.sample_count += int(dose_values_gy.size)

    def finalize_curve(
        self,
        name: str,
        color_rgb: Tuple[int, int, int],
        oversampling_factor: float,
        used_fractional_labelmap: bool,
        metadata: Optional[Dict[str, float | str]] = None,
    ) -> Optional[DVHCurve]:
        if self.total_volume_cc <= 0.0:
            return None

        cumulative_cc = np.cumsum(self.hist_cc[::-1])[::-1].astype(np.float64)
        volume_pct = cumulative_cc / self.total_volume_cc * 100.0
        dose_axis = np.concatenate([self.bin_edges_gy[:-1], [self.bin_edges_gy[-1]]]).astype(np.float32)
        volume_pct_axis = np.concatenate([volume_pct, [0.0]]).astype(np.float32)
        volume_cc_axis = np.concatenate([cumulative_cc, [0.0]]).astype(np.float32)

        if not np.isfinite(self.min_dose_gy):
            self.min_dose_gy = 0.0
        if not np.isfinite(self.max_dose_gy):
            self.max_dose_gy = 0.0

        return DVHCurve(
            name=name,
            color_rgb=color_rgb,
            dose_bins_gy=dose_axis,
            volume_pct=volume_pct_axis,
            voxel_count=self.sample_count,
            volume_cc=float(self.total_volume_cc),
            mean_dose_gy=float(self.weighted_dose_sum_gy_cc / self.total_volume_cc),
            max_dose_gy=float(self.max_dose_gy),
            min_dose_gy=float(self.min_dose_gy),
            volume_cc_axis=volume_cc_axis,
            oversampling_factor=float(oversampling_factor),
            used_fractional_labelmap=bool(used_fractional_labelmap),
            metadata=dict(metadata or {}),
        )


# -----------------------------------------------------------------------------
# DVH metric helpers
# -----------------------------------------------------------------------------


def _volume_cc_axis(curve: DVHCurve) -> np.ndarray:
    if getattr(curve, "volume_cc_axis", None) is not None and curve.volume_cc_axis.size == curve.dose_bins_gy.size:
        return curve.volume_cc_axis.astype(np.float64)
    return curve.volume_pct.astype(np.float64) * float(curve.volume_cc) / 100.0


def volume_pct_at_dose_gy(curve: DVHCurve, dose_gy: float) -> float:
    if curve.dose_bins_gy.size == 0 or curve.volume_pct.size == 0:
        return 0.0
    return float(
        np.interp(
            float(dose_gy),
            curve.dose_bins_gy.astype(np.float64),
            curve.volume_pct.astype(np.float64),
            left=float(curve.volume_pct[0]),
            right=float(curve.volume_pct[-1]),
        )
    )


def volume_cc_at_dose_gy(curve: DVHCurve, dose_gy: float) -> float:
    if curve.dose_bins_gy.size == 0:
        return 0.0
    return float(
        np.interp(
            float(dose_gy),
            curve.dose_bins_gy.astype(np.float64),
            _volume_cc_axis(curve),
            left=float(_volume_cc_axis(curve)[0]),
            right=float(_volume_cc_axis(curve)[-1]),
        )
    )


def dose_at_volume_pct(curve: DVHCurve, volume_pct: float) -> float:
    if curve.dose_bins_gy.size == 0 or curve.volume_pct.size == 0:
        return 0.0
    target = float(np.clip(volume_pct, 0.0, 100.0))
    return float(
        np.interp(
            target,
            curve.volume_pct[::-1].astype(np.float64),
            curve.dose_bins_gy[::-1].astype(np.float64),
        )
    )


def dose_at_volume_cc(curve: DVHCurve, volume_cc: float) -> float:
    if curve.dose_bins_gy.size == 0:
        return 0.0
    volume_axis = _volume_cc_axis(curve)
    target = float(np.clip(volume_cc, 0.0, max(float(curve.volume_cc), 0.0)))
    return float(
        np.interp(
            target,
            volume_axis[::-1],
            curve.dose_bins_gy[::-1].astype(np.float64),
        )
    )


# -----------------------------------------------------------------------------
# Public helpers to build DVHs from weighted samples
# -----------------------------------------------------------------------------


def _default_bin_edges(max_dose_gy: float, bin_width_gy: float) -> np.ndarray:
    max_dose_gy = max(float(max_dose_gy), float(bin_width_gy))
    bin_width_gy = max(float(bin_width_gy), 1e-6)
    n_bins = int(np.ceil(max_dose_gy / bin_width_gy))
    return np.linspace(0.0, n_bins * bin_width_gy, n_bins + 1, dtype=np.float64)


def build_dvh_curve_from_weighted_samples(
    name: str,
    color_rgb: Tuple[int, int, int],
    dose_values_gy: np.ndarray,
    weights_cc: np.ndarray,
    *,
    bin_width_gy: float = 0.2,
    max_dose_gy: Optional[float] = None,
    oversampling_factor: float = 1.0,
    used_fractional_labelmap: bool = False,
    metadata: Optional[Dict[str, float | str]] = None,
) -> Optional[DVHCurve]:
    dose_values_gy = np.asarray(dose_values_gy, dtype=np.float64)
    weights_cc = np.asarray(weights_cc, dtype=np.float64)
    if dose_values_gy.size == 0 or weights_cc.size == 0:
        return None

    if max_dose_gy is None:
        max_dose_gy = float(np.max(dose_values_gy))
    edges = _default_bin_edges(max_dose_gy, bin_width_gy)
    acc = _WeightedHistogramAccumulator(bin_edges_gy=edges, hist_cc=np.zeros(edges.size - 1, dtype=np.float64))
    acc.add(dose_values_gy, weights_cc)
    return acc.finalize_curve(
        name=name,
        color_rgb=color_rgb,
        oversampling_factor=oversampling_factor,
        used_fractional_labelmap=used_fractional_labelmap,
        metadata=metadata,
    )


# -----------------------------------------------------------------------------
# Main DVH computation for RT structures
# -----------------------------------------------------------------------------


def _iter_chunks(indices: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    if indices.size == 0:
        return []
    return [indices[i : i + chunk_size] for i in range(0, indices.size, chunk_size)]


def _build_local_occupancy_grid(
    contours_rc: Sequence[np.ndarray],
    scale: float,
    *,
    use_fractional_labelmap: bool,
    fractional_subdivisions: int,
    max_border_batch_points: int,
) -> Tuple[np.ndarray, int, int]:
    mask, row_offset, col_offset = _build_scaled_local_mask(contours_rc, scale)
    if mask.size == 0 or not np.any(mask):
        return np.zeros((0, 0), dtype=np.float32), 0, 0

    occupancy = np.zeros(mask.shape, dtype=np.float32)
    interior_mask, border_mask = _mask_interior_border(mask)
    occupancy[interior_mask] = 1.0

    if not np.any(border_mask):
        return occupancy, row_offset, col_offset

    border_r, border_c = np.where(border_mask)
    if border_r.size == 0:
        return occupancy, row_offset, col_offset

    if not use_fractional_labelmap or fractional_subdivisions <= 1:
        occupancy[border_r, border_c] = 1.0
        return occupancy, row_offset, col_offset

    frac_div = int(max(1, fractional_subdivisions))
    sub_offsets = (np.arange(frac_div, dtype=np.float32) + 0.5) / frac_div
    border_indices = np.arange(border_r.size, dtype=np.int64)
    cell_batch = max(1, max_border_batch_points // max(frac_div * frac_div, 1))

    for chunk in _iter_chunks(border_indices, cell_batch):
        base_rows = np.float32(row_offset) + border_r[chunk].astype(np.float32)
        base_cols = np.float32(col_offset) + border_c[chunk].astype(np.float32)

        rr = (base_rows[:, None, None] + sub_offsets[None, :, None]).repeat(frac_div, axis=2)
        cc = (base_cols[:, None, None] + sub_offsets[None, None, :]).repeat(frac_div, axis=1)
        inside = _points_in_contours_xor(
            rr.reshape(-1) / scale,
            cc.reshape(-1) / scale,
            contours_rc,
        ).reshape(-1, frac_div, frac_div)
        occupancy[border_r[chunk], border_c[chunk]] = np.mean(inside, axis=(1, 2), dtype=np.float32)

    return occupancy, row_offset, col_offset


def _accumulate_structure_slice(
    acc: _WeightedHistogramAccumulator,
    dose: DoseVolume,
    dose_context: _DoseSamplingContext,
    contours_dose_rc: Sequence[np.ndarray],
    plane_position_mm: float,
    slab_thickness_mm: float,
    oversampling_factor: float,
    options: DVHCalculationOptions,
    z_offset_cache: Dict[Tuple[float, float], np.ndarray],
) -> None:
    if not contours_dose_rc or slab_thickness_mm <= 0.0:
        return

    scale = float(max(options.minimum_oversampling_factor, min(options.maximum_oversampling_factor, oversampling_factor)))
    occupancy, row_offset, col_offset = _build_local_occupancy_grid(
        contours_dose_rc,
        scale,
        use_fractional_labelmap=options.use_fractional_labelmap,
        fractional_subdivisions=options.fractional_subdivisions,
        max_border_batch_points=options.max_border_batch_points,
    )
    if occupancy.size == 0:
        return

    occupied_r, occupied_c = np.where(occupancy > 0.0)
    if occupied_r.size == 0:
        return

    z_key = (round(float(slab_thickness_mm), 4), round(float(oversampling_factor), 4))
    z_offsets_mm = z_offset_cache.get(z_key)
    if z_offsets_mm is None:
        z_offsets_mm = _dose_z_sampling_offsets_mm(slab_thickness_mm, oversampling_factor)
        z_offset_cache[z_key] = z_offsets_mm
    z_count = max(int(z_offsets_mm.size), 1)

    cell_area_mm2 = np.float32(dose_context.spacing_col_mm * dose_context.spacing_row_mm / (scale * scale))
    voxel_weights_cc = (
        occupancy[occupied_r, occupied_c].astype(np.float32, copy=False)
        * cell_area_mm2
        * np.float32(slab_thickness_mm)
        / np.float32(1000.0 * z_count)
    ).astype(np.float32, copy=False)
    row_coords = ((np.float32(row_offset) + occupied_r.astype(np.float32) + 0.5) / scale).astype(np.float32, copy=False)
    col_coords = ((np.float32(col_offset) + occupied_c.astype(np.float32) + 0.5) / scale).astype(np.float32, copy=False)

    for z_offset_mm in z_offsets_mm:
        dose_values = _sample_dose_plane_virtual_rc(
            dose,
            row_coords,
            col_coords,
            plane_position_mm + float(z_offset_mm),
            linear_interpolation=options.use_linear_dose_interpolation,
            dose_context=dose_context,
        )
        acc.add(dose_values, voxel_weights_cc)


def compute_dvh_curves(
    ct: CTVolume,
    dose: DoseVolume,
    rtstruct: RTStructData,
    *,
    options: DVHCalculationOptions = DEFAULT_DVH_OPTIONS,
    mask_cache: Optional[List[Dict[int, np.ndarray]]] = None,
) -> List[DVHCurve]:
    if ct is None or dose is None or rtstruct is None:
        return []

    mask_cache = _ensure_mask_cache(rtstruct, ct.rows, ct.cols, mask_cache)
    max_dose_gy = float(options.max_dose_gy) if options.max_dose_gy is not None else float(np.nanmax(dose.dose_gy))
    bin_edges = _default_bin_edges(max_dose_gy, options.dose_bin_width_gy)
    slice_thicknesses = _slice_thicknesses_mm(ct.z_positions_mm, float(ct.spacing_xyz_mm[2]))
    dose_context = _get_dose_sampling_context(dose)
    ct_context = _get_ct_geometry_context(ct)
    rtstruct_cache = _get_runtime_cache(rtstruct, "_peer_dvh_runtime_cache")
    geometry_key = (id(ct), id(dose))
    decision_key = (
        geometry_key,
        bool(options.automatic_oversampling),
        float(options.fixed_oversampling_factor),
        float(options.minimum_oversampling_factor),
        float(options.maximum_oversampling_factor),
    )
    metrics_cache = rtstruct_cache.setdefault(("metrics", geometry_key), {})
    decision_cache = rtstruct_cache.setdefault(("decision", decision_key), {})
    slice_transform_cache = rtstruct_cache.setdefault(("slice_transform", geometry_key), {})
    contour_transform_cache = rtstruct_cache.setdefault(("contours_dose_rc", geometry_key), {})
    z_offset_cache: Dict[Tuple[float, float], np.ndarray] = {}

    curves: List[DVHCurve] = []

    for structure_index, structure in enumerate(rtstruct.structures):
        metrics = metrics_cache.get(structure_index)
        if metrics is None:
            metrics = estimate_structure_geometry_metrics(
                ct,
                dose,
                structure,
                structure_mask_cache=mask_cache[structure_index] if structure_index < len(mask_cache) else None,
            )
            metrics_cache[structure_index] = metrics

        decision = decision_cache.get(structure_index)
        if decision is None:
            decision = (
                compute_oversampling_factor_from_metrics(metrics.rss, metrics.complexity, options)
                if options.automatic_oversampling
                else OversamplingDecision(
                    oversampling_factor=float(options.fixed_oversampling_factor),
                    power_of_two=float(np.log2(max(options.fixed_oversampling_factor, 1e-6))),
                    rss=metrics.rss,
                    complexity=metrics.complexity,
                )
            )
            decision_cache[structure_index] = decision

        accumulator = _WeightedHistogramAccumulator(
            bin_edges_gy=bin_edges,
            hist_cc=np.zeros(bin_edges.size - 1, dtype=np.float64),
        )

        for slice_index in sorted(structure.points_rc_by_slice):
            contours_rc = structure.points_rc_by_slice.get(slice_index, [])
            if not contours_rc:
                continue
            slab_thickness_mm = (
                float(slice_thicknesses[slice_index]) if slice_index < slice_thicknesses.size else float(ct.spacing_xyz_mm[2])
            )

            slice_transform = slice_transform_cache.get(slice_index)
            if slice_transform is None:
                slice_transform = _build_ct_slice_to_dose_transform(ct_context, dose_context, slice_index)
                slice_transform_cache[slice_index] = slice_transform
            contour_cache_key = (structure_index, slice_index)
            contours_dose_rc = contour_transform_cache.get(contour_cache_key)
            if contours_dose_rc is None:
                contours_dose_rc = [
                    (
                        np.asarray(contour, dtype=np.float32) @ slice_transform.transform_rc_to_dose_rc.T
                        + slice_transform.offset_rc_to_dose_rc[None, :]
                    ).astype(np.float32, copy=False)
                    for contour in contours_rc
                ]
                contour_transform_cache[contour_cache_key] = contours_dose_rc

            _accumulate_structure_slice(
                accumulator,
                dose,
                dose_context,
                contours_dose_rc,
                slice_transform.plane_position_mm,
                slab_thickness_mm,
                decision.oversampling_factor,
                options,
                z_offset_cache,
            )

        curve = accumulator.finalize_curve(
            name=structure.name,
            color_rgb=structure.color_rgb,
            oversampling_factor=decision.oversampling_factor,
            used_fractional_labelmap=options.use_fractional_labelmap,
            metadata={
                "rss": metrics.rss,
                "complexity": metrics.complexity,
                "surface_area_mm2": metrics.surface_area_mm2,
                "volume_mm3": metrics.volume_mm3,
                "oversampling_power": decision.power_of_two,
            },
        )
        if curve is not None:
            curves.append(curve)

    return curves


__all__ = [
    "DVHCalculationOptions",
    "DEFAULT_DVH_OPTIONS",
    "OversamplingDecision",
    "StructureGeometryMetrics",
    "build_dvh_curve_from_weighted_samples",
    "compute_dvh_curves",
    "compute_oversampling_factor_from_metrics",
    "dose_at_volume_cc",
    "dose_at_volume_pct",
    "estimate_structure_geometry_metrics",
    "trilinear_sample_dose_patient_xyz",
    "volume_cc_at_dose_gy",
    "volume_pct_at_dose_gy",
]
