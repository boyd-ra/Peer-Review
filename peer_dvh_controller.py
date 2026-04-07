from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from peer_helpers import normalize_structure_name, volume_cc_at_dose_gy, volume_pct_at_dose_gy
from peer_models import DVHCurve


@dataclass(frozen=True)
class DvhReadoutState:
    dose_gy: float
    volume_pct: float
    volume_cc: float
    text: str


def get_curve_for_name(curves: Sequence[DVHCurve], normalized_name: str) -> Optional[DVHCurve]:
    for curve in curves:
        if normalize_structure_name(curve.name) == normalized_name:
            return curve
    return None


def get_current_curve_names(curves: Sequence[DVHCurve]) -> List[str]:
    return [normalize_structure_name(curve.name) for curve in curves]


def get_visible_dvh_curves(
    curves: Sequence[DVHCurve],
    visibility_resolver: Callable[[str], bool],
) -> List[DVHCurve]:
    return [
        curve
        for curve in curves
        if visibility_resolver(normalize_structure_name(curve.name))
    ]


def get_dvh_plot_arrays(curve: DVHCurve) -> Tuple[np.ndarray, np.ndarray]:
    plot_dose_bins = curve.dose_bins_gy
    plot_volume_pct = curve.volume_pct
    if plot_dose_bins.size <= 1 or plot_volume_pct.size != plot_dose_bins.size:
        return plot_dose_bins, plot_volume_pct
    zero_indices = np.flatnonzero(plot_volume_pct.astype(np.float64) <= 1e-6)
    if zero_indices.size == 0:
        return plot_dose_bins, plot_volume_pct
    end_index = max(int(zero_indices[0]) + 1, 2)
    return (
        plot_dose_bins[:end_index].astype(np.float32, copy=False),
        plot_volume_pct[:end_index].astype(np.float32, copy=False),
    )


def get_visible_dvh_view_range(
    curves: Sequence[DVHCurve],
    visibility_resolver: Callable[[str], bool],
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    visible_curves = [
        curve for curve in get_visible_dvh_curves(curves, visibility_resolver) if curve.dose_bins_gy.size
    ]
    if not visible_curves:
        return None

    max_dose: Optional[float] = None
    for curve in visible_curves:
        plot_dose_bins, _plot_volume_pct = get_dvh_plot_arrays(curve)
        if plot_dose_bins.size == 0:
            continue
        curve_max_dose = float(plot_dose_bins[-1])
        max_dose = curve_max_dose if max_dose is None else max(max_dose, curve_max_dose)

    if max_dose is None:
        return None

    return ((0.0, max(max_dose, 1.0)), (0.0, 100.0))


def find_nearest_dvh_curve_name(
    curves: Sequence[DVHCurve],
    plot_item: pg.PlotItem,
    scene_pos: QtCore.QPointF,
    tolerance_px: float = 16.0,
) -> Optional[str]:
    if not curves:
        return None

    if not plot_item.vb.sceneBoundingRect().contains(scene_pos):
        return None

    view_pos = plot_item.vb.mapSceneToView(scene_pos)
    clicked_dose = float(view_pos.x())
    clicked_volume = float(view_pos.y())
    x_range, y_range = plot_item.vb.viewRange()
    x_span = max(float(x_range[1] - x_range[0]), 1e-6)
    y_span = max(float(y_range[1] - y_range[0]), 1e-6)
    best_name: Optional[str] = None
    best_distance_px: Optional[float] = None
    best_normalized_score: Optional[float] = None

    for curve in curves:
        if curve.dose_bins_gy.size == 0:
            continue
        plot_dose_bins, _plot_volume_pct = get_dvh_plot_arrays(curve)
        if plot_dose_bins.size == 0:
            continue
        clamped_dose = float(np.clip(clicked_dose, float(plot_dose_bins[0]), float(plot_dose_bins[-1])))
        volume_pct = float(volume_pct_at_dose_gy(curve, clamped_dose))
        curve_scene_pos = plot_item.vb.mapViewToScene(QtCore.QPointF(clamped_dose, volume_pct))
        distance_px = float(QtCore.QLineF(scene_pos, curve_scene_pos).length())
        dose_distance = abs(clamped_dose - clicked_dose) / x_span
        volume_distance = abs(volume_pct - clicked_volume) / y_span
        normalized_score = volume_distance + 0.35 * dose_distance
        if best_distance_px is None or distance_px < best_distance_px:
            best_distance_px = distance_px
        if best_normalized_score is None or normalized_score < best_normalized_score:
            best_normalized_score = normalized_score
            best_name = normalize_structure_name(curve.name)

    if best_distance_px is not None and best_distance_px <= tolerance_px:
        return best_name
    if best_normalized_score is not None and best_normalized_score <= 0.15:
        return best_name
    return None


def build_dvh_readout_state(
    curve: DVHCurve,
    plot_item: pg.PlotItem,
    scene_pos: QtCore.QPointF,
) -> Optional[DvhReadoutState]:
    if curve.dose_bins_gy.size == 0:
        return None
    if not plot_item.vb.sceneBoundingRect().contains(scene_pos):
        return None

    plot_dose_bins, _plot_volume_pct = get_dvh_plot_arrays(curve)
    if plot_dose_bins.size == 0:
        return None

    view_pos = plot_item.vb.mapSceneToView(scene_pos)
    dose_gy = float(np.clip(view_pos.x(), float(plot_dose_bins[0]), float(plot_dose_bins[-1])))
    volume_pct = float(volume_pct_at_dose_gy(curve, dose_gy))
    volume_cc = float(volume_cc_at_dose_gy(curve, dose_gy))
    return DvhReadoutState(
        dose_gy=dose_gy,
        volume_pct=volume_pct,
        volume_cc=volume_cc,
        text=f"{curve.name}: Dose {dose_gy:.2f} Gy | Volume {volume_pct:.1f}% ({volume_cc:.2f} cc)",
    )
