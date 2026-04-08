from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from peer_helpers import (
    build_outline_series,
    dose_to_rgba,
    line_intersections_at_col,
    line_intersections_at_row,
    normalize_structure_name,
    orthogonal_row_scale,
    resample_orthogonal_plane,
    sample_dose_to_ct_slice,
)
from peer_models import CTVolume, DoseVolume, RTStructData


@dataclass(frozen=True)
class PolylineSpec:
    x: np.ndarray
    y: np.ndarray
    color_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class OrthogonalRenderState:
    sagittal_plane: np.ndarray
    coronal_plane: np.ndarray
    sagittal_scale: float
    coronal_scale: float
    sagittal_dose_plane: Optional[np.ndarray]
    coronal_dose_plane: Optional[np.ndarray]
    sagittal_dose_rgba: np.ndarray
    coronal_dose_rgba: np.ndarray
    sagittal_contours: List[PolylineSpec]
    coronal_contours: List[PolylineSpec]


@dataclass(frozen=True)
class AxialRenderState:
    ct_plane: np.ndarray
    dose_plane: Optional[np.ndarray]
    dose_rgba: np.ndarray
    contour_specs: List[PolylineSpec]
    slice_label_text: str
    z_label_text: str
    window_label_text: str


@dataclass(frozen=True)
class MaxDoseMarkerState:
    axial_point: Optional[Tuple[float, float]]
    sagittal_point: Optional[Tuple[float, float]]
    coronal_point: Optional[Tuple[float, float]]


@dataclass(frozen=True)
class MaxDoseCenterPoints:
    axial_point: Tuple[float, float]
    sagittal_point: Tuple[float, float]
    coronal_point: Tuple[float, float]


@dataclass(frozen=True)
class AxialOverlayPositions:
    autoscroll_pos: Tuple[int, int]
    readout_pos: Tuple[int, int]


def rotate_plane_180(plane: np.ndarray) -> np.ndarray:
    if plane.size == 0:
        return plane
    return np.flip(plane, axis=(0, 1)).copy()


def rotate_polyline_specs_180(
    specs: Sequence[PolylineSpec],
    *,
    width: int,
    height: int,
) -> List[PolylineSpec]:
    if width <= 0 or height <= 0:
        return list(specs)
    rotated_specs: List[PolylineSpec] = []
    width_max = float(width - 1)
    height_max = float(height - 1)
    for spec in specs:
        rotated_specs.append(
            PolylineSpec(
                x=np.asarray(width_max - spec.x, dtype=np.float32),
                y=np.asarray(height_max - spec.y, dtype=np.float32),
                color_rgb=spec.color_rgb,
            )
        )
    return rotated_specs


def rotate_point_180(
    point: Optional[Tuple[float, float]],
    *,
    width: int,
    height: int,
) -> Optional[Tuple[float, float]]:
    if point is None or width <= 0 or height <= 0:
        return point
    return (float(width - 1) - float(point[0]), float(height - 1) - float(point[1]))


def build_closed_contour_spec(contour_rc: np.ndarray, color_rgb: Tuple[int, int, int]) -> Optional[PolylineSpec]:
    rr = np.asarray(contour_rc[:, 0], dtype=np.float32)
    cc = np.asarray(contour_rc[:, 1], dtype=np.float32)
    if rr.size == 0 or cc.size == 0:
        return None
    if rr.size >= 2 and cc.size >= 2:
        first_row = float(rr[0])
        first_col = float(cc[0])
        if not (np.isclose(float(rr[-1]), first_row) and np.isclose(float(cc[-1]), first_col)):
            rr = np.append(rr, np.float32(first_row))
            cc = np.append(cc, np.float32(first_col))
    return PolylineSpec(x=cc, y=rr, color_rgb=color_rgb)


def build_axial_contour_specs(
    rtstruct: Optional[RTStructData],
    slice_index: int,
    structure_visibility_resolver: Callable[[int], bool],
) -> List[PolylineSpec]:
    if rtstruct is None:
        return []

    specs: List[PolylineSpec] = []
    for idx, structure in enumerate(rtstruct.structures):
        if not structure_visibility_resolver(idx):
            continue
        for contour_rc in structure.points_rc_by_slice.get(slice_index, []):
            spec = build_closed_contour_spec(contour_rc, structure.color_rgb)
            if spec is not None:
                specs.append(spec)
    return specs


def build_orthogonal_contour_specs(
    rtstruct: Optional[RTStructData],
    row_idx: int,
    col_idx: int,
    sagittal_scale: float,
    coronal_scale: float,
    structure_visibility_resolver: Callable[[int], bool],
) -> Tuple[List[PolylineSpec], List[PolylineSpec]]:
    if rtstruct is None:
        return [], []

    sagittal_specs: List[PolylineSpec] = []
    coronal_specs: List[PolylineSpec] = []
    for idx, structure in enumerate(rtstruct.structures):
        if not structure_visibility_resolver(idx):
            continue
        sagittal_samples: dict[int, List[float]] = {}
        coronal_samples: dict[int, List[float]] = {}
        for slice_index, contours in structure.points_rc_by_slice.items():
            for contour_rc in contours:
                sagittal_intersections = line_intersections_at_col(contour_rc, float(col_idx))
                if sagittal_intersections:
                    sagittal_samples.setdefault(slice_index, []).extend(sagittal_intersections)

                coronal_intersections = line_intersections_at_row(contour_rc, float(row_idx))
                if coronal_intersections:
                    coronal_samples.setdefault(slice_index, []).extend(coronal_intersections)

        for slice_index in sagittal_samples:
            sagittal_samples[slice_index].sort()
        for slice_index in coronal_samples:
            coronal_samples[slice_index].sort()

        for xs, ys in build_outline_series(sagittal_samples, sagittal_scale):
            sagittal_specs.append(
                PolylineSpec(
                    x=np.asarray(xs, dtype=np.float32),
                    y=np.asarray(ys, dtype=np.float32),
                    color_rgb=structure.color_rgb,
                )
            )

        for xs, ys in build_outline_series(coronal_samples, coronal_scale):
            coronal_specs.append(
                PolylineSpec(
                    x=np.asarray(xs, dtype=np.float32),
                    y=np.asarray(ys, dtype=np.float32),
                    color_rgb=structure.color_rgb,
                )
            )

    return sagittal_specs, coronal_specs


def build_orthogonal_render_state(
    ct: CTVolume,
    rtstruct: Optional[RTStructData],
    sampled_dose_volume_ct: Optional[np.ndarray],
    row_idx: int,
    col_idx: int,
    lo: float,
    hi: float,
    dose_alpha: float,
    min_dose: float,
    max_dose: float,
    structure_visibility_resolver: Callable[[int], bool],
) -> OrthogonalRenderState:
    sx = float(ct.spacing_xyz_mm[0])
    sy = float(ct.spacing_xyz_mm[1])
    sz = float(ct.spacing_xyz_mm[2])
    sagittal_scale = orthogonal_row_scale(sz, sy)
    coronal_scale = orthogonal_row_scale(sz, sx)

    sagittal_plane = resample_orthogonal_plane(ct.volume_hu[:, :, col_idx], sz, sy)
    coronal_plane = resample_orthogonal_plane(ct.volume_hu[:, row_idx, :], sz, sx)

    sagittal_dose_plane: Optional[np.ndarray] = None
    coronal_dose_plane: Optional[np.ndarray] = None
    sagittal_dose_rgba = np.zeros(sagittal_plane.shape + (4,), dtype=np.uint8)
    coronal_dose_rgba = np.zeros(coronal_plane.shape + (4,), dtype=np.uint8)
    if sampled_dose_volume_ct is not None:
        sagittal_dose_plane = resample_orthogonal_plane(sampled_dose_volume_ct[:, :, col_idx], sz, sy)
        coronal_dose_plane = resample_orthogonal_plane(sampled_dose_volume_ct[:, row_idx, :], sz, sx)
        sagittal_dose_rgba = dose_to_rgba(
            sagittal_dose_plane,
            alpha=dose_alpha,
            min_dose_gy=min_dose,
            max_dose_gy=max_dose,
        )
        coronal_dose_rgba = dose_to_rgba(
            coronal_dose_plane,
            alpha=dose_alpha,
            min_dose_gy=min_dose,
            max_dose_gy=max_dose,
        )

    sagittal_contours, coronal_contours = build_orthogonal_contour_specs(
        rtstruct,
        row_idx,
        col_idx,
        sagittal_scale,
        coronal_scale,
        structure_visibility_resolver,
    )

    sagittal_height, sagittal_width = sagittal_plane.shape[:2]
    coronal_height, coronal_width = coronal_plane.shape[:2]
    sagittal_plane = rotate_plane_180(sagittal_plane)
    coronal_plane = rotate_plane_180(coronal_plane)
    sagittal_dose_rgba = rotate_plane_180(sagittal_dose_rgba)
    coronal_dose_rgba = rotate_plane_180(coronal_dose_rgba)
    if sagittal_dose_plane is not None:
        sagittal_dose_plane = rotate_plane_180(sagittal_dose_plane)
    if coronal_dose_plane is not None:
        coronal_dose_plane = rotate_plane_180(coronal_dose_plane)
    sagittal_contours = rotate_polyline_specs_180(
        sagittal_contours,
        width=sagittal_width,
        height=sagittal_height,
    )
    coronal_contours = rotate_polyline_specs_180(
        coronal_contours,
        width=coronal_width,
        height=coronal_height,
    )

    return OrthogonalRenderState(
        sagittal_plane=sagittal_plane,
        coronal_plane=coronal_plane,
        sagittal_scale=sagittal_scale,
        coronal_scale=coronal_scale,
        sagittal_dose_plane=sagittal_dose_plane,
        coronal_dose_plane=coronal_dose_plane,
        sagittal_dose_rgba=sagittal_dose_rgba,
        coronal_dose_rgba=coronal_dose_rgba,
        sagittal_contours=sagittal_contours,
        coronal_contours=coronal_contours,
    )


def build_axial_render_state(
    ct: CTVolume,
    dose: Optional[DoseVolume],
    rtstruct: Optional[RTStructData],
    sampled_dose_volume_ct: Optional[np.ndarray],
    slice_index: int,
    lo: float,
    hi: float,
    dose_alpha: float,
    min_dose: float,
    max_dose: float,
    structure_visibility_resolver: Callable[[int], bool],
) -> AxialRenderState:
    ct_plane = ct.volume_hu[slice_index]

    dose_plane: Optional[np.ndarray] = None
    if dose is not None:
        if sampled_dose_volume_ct is not None:
            dose_plane = sampled_dose_volume_ct[slice_index]
        else:
            dose_plane = sample_dose_to_ct_slice(ct, dose, slice_index)

    if dose_plane is None:
        dose_rgba = np.zeros((ct.rows, ct.cols, 4), dtype=np.uint8)
    else:
        dose_rgba = dose_to_rgba(
            dose_plane,
            alpha=dose_alpha,
            min_dose_gy=min_dose,
            max_dose_gy=max_dose,
        )

    contour_specs = build_axial_contour_specs(rtstruct, slice_index, structure_visibility_resolver)
    return AxialRenderState(
        ct_plane=ct_plane,
        dose_plane=dose_plane,
        dose_rgba=dose_rgba,
        contour_specs=contour_specs,
        slice_label_text=f"Slice: {slice_index + 1}/{ct.volume_hu.shape[0]}",
        z_label_text=f"Plane pos: {ct.z_positions_mm[slice_index]:.2f}",
        window_label_text=f"WL/WW: {int((lo + hi) / 2.0)} / {int(hi - lo)}",
    )


def build_axial_hover_text(
    ct: Optional[CTVolume],
    displayed_dose_plane: Optional[np.ndarray],
    slice_index: int,
    row_idx: int,
    col_idx: int,
) -> Optional[str]:
    if ct is None:
        return None
    if not (0 <= row_idx < ct.rows and 0 <= col_idx < ct.cols):
        return None

    hu = float(ct.volume_hu[slice_index, row_idx, col_idx])
    text = f"HU {hu:.1f}"
    if displayed_dose_plane is not None:
        dose_gy = float(displayed_dose_plane[row_idx, col_idx])
        text += f"\nDose {dose_gy:.2f} Gy"
    return text


def clear_overlay_items(view: pg.ViewBox, items: List[pg.GraphicsObject]) -> None:
    for item in items:
        view.removeItem(item)
    items.clear()


def apply_polyline_specs(
    view: pg.ViewBox,
    target_items: List[pg.PlotCurveItem],
    specs: Sequence[PolylineSpec],
    *,
    width: int = 4,
    z_value: float = 2.0,
) -> None:
    clear_overlay_items(view, target_items)
    for spec in specs:
        curve = pg.PlotCurveItem(
            x=spec.x,
            y=spec.y,
            pen=pg.mkPen(color=spec.color_rgb, width=width),
        )
        curve.setZValue(z_value)
        view.addItem(curve)
        target_items.append(curve)


def apply_isodose_items(
    view: pg.ViewBox,
    target_items: List[pg.IsocurveItem],
    dose_plane: Optional[np.ndarray],
    active_levels: Sequence[Tuple[float, Tuple[int, int, int]]],
) -> None:
    clear_overlay_items(view, target_items)
    if dose_plane is None or not np.isfinite(dose_plane).any():
        return

    max_dose = float(np.nanmax(dose_plane))
    if max_dose <= 0.0:
        return

    for dose_gy, color_rgb in active_levels:
        if dose_gy <= 0.0 or dose_gy > max_dose:
            continue
        pen = pg.mkPen(color=color_rgb, width=3, style=QtCore.Qt.PenStyle.DashLine)
        item = pg.IsocurveItem(
            data=dose_plane,
            level=dose_gy,
            pen=pen,
            axisOrder="row-major",
        )
        item.setZValue(1.5)
        view.addItem(item)
        target_items.append(item)


def get_orthogonal_scales(ct: CTVolume) -> Tuple[float, float]:
    sx = float(ct.spacing_xyz_mm[0])
    sy = float(ct.spacing_xyz_mm[1])
    sz = float(ct.spacing_xyz_mm[2])
    return orthogonal_row_scale(sz, sy), orthogonal_row_scale(sz, sx)


def get_orthogonal_display_sizes(ct: CTVolume) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    sagittal_scale, coronal_scale = get_orthogonal_scales(ct)
    depth = int(ct.volume_hu.shape[0])
    sagittal_height = max(1, int(round(depth * sagittal_scale)))
    coronal_height = max(1, int(round(depth * coronal_scale)))
    sagittal_width = int(ct.rows)
    coronal_width = int(ct.cols)
    return (sagittal_width, sagittal_height), (coronal_width, coronal_height)


def build_active_isodose_levels(
    texts: Sequence[str],
    colors: Sequence[Tuple[int, int, int]],
) -> List[Tuple[float, Tuple[int, int, int]]]:
    levels: List[Tuple[float, Tuple[int, int, int]]] = []
    seen: set[float] = set()
    for color_rgb, text in zip(colors, texts):
        stripped = text.strip()
        if not stripped:
            continue
        try:
            dose_gy = float(stripped)
        except ValueError:
            continue
        if dose_gy <= 0.0:
            continue
        rounded = round(dose_gy, 3)
        if rounded in seen:
            continue
        seen.add(rounded)
        levels.append((dose_gy, color_rgb))
    return levels


def resolve_axial_indices(
    ct: Optional[CTVolume],
    x: float,
    y: float,
) -> Optional[Tuple[int, int]]:
    if ct is None:
        return None
    col_idx = int(round(x))
    row_idx = int(round(y))
    if not (0 <= row_idx < ct.rows and 0 <= col_idx < ct.cols):
        return None
    return row_idx, col_idx


def build_max_dose_marker_state(
    ct: Optional[CTVolume],
    max_dose_index_zyx: Optional[Tuple[int, int, int]],
    slice_index: int,
    current_row: int,
    current_col: int,
) -> MaxDoseMarkerState:
    if ct is None or max_dose_index_zyx is None:
        return MaxDoseMarkerState(None, None, None)

    k, r, c = max_dose_index_zyx
    sagittal_scale, coronal_scale = get_orthogonal_scales(ct)
    (sagittal_width, sagittal_height), (coronal_width, coronal_height) = get_orthogonal_display_sizes(ct)

    axial_point = (float(c), float(r)) if slice_index == k else None
    sagittal_point = rotate_point_180(
        (
            (float(r), float(k) * sagittal_scale)
            if int(np.clip(current_col, 0, ct.cols - 1)) == c
            else None
        ),
        width=sagittal_width,
        height=sagittal_height,
    )
    coronal_point = rotate_point_180(
        (
            (float(c), float(k) * coronal_scale)
            if int(np.clip(current_row, 0, ct.rows - 1)) == r
            else None
        ),
        width=coronal_width,
        height=coronal_height,
    )
    return MaxDoseMarkerState(axial_point, sagittal_point, coronal_point)


def build_max_dose_center_points(
    ct: Optional[CTVolume],
    max_dose_index_zyx: Optional[Tuple[int, int, int]],
) -> Optional[MaxDoseCenterPoints]:
    if ct is None or max_dose_index_zyx is None:
        return None
    k, r, c = max_dose_index_zyx
    sagittal_scale, coronal_scale = get_orthogonal_scales(ct)
    (sagittal_width, sagittal_height), (coronal_width, coronal_height) = get_orthogonal_display_sizes(ct)
    return MaxDoseCenterPoints(
        axial_point=(float(c), float(r)),
        sagittal_point=rotate_point_180(
            (float(r), float(k) * sagittal_scale),
            width=sagittal_width,
            height=sagittal_height,
        ),
        coronal_point=rotate_point_180(
            (float(c), float(k) * coronal_scale),
            width=coronal_width,
            height=coronal_height,
        ),
    )


def build_axial_overlay_positions(
    graphics_width: int,
    readout_width: int,
    margin: int = 10,
) -> AxialOverlayPositions:
    return AxialOverlayPositions(
        autoscroll_pos=(margin, margin),
        readout_pos=(max(margin, graphics_width - readout_width - margin), margin),
    )
