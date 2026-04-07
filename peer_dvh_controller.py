from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from peer_helpers import normalize_structure_name, volume_cc_at_dose_gy, volume_pct_at_dose_gy
from peer_models import DVHCurve, RTStructData, StructureGoal, StructureGoalEvaluation
from peer_viewer_support import evaluate_visible_structure_goals


DVH_MISSING_INPUTS_STATUS_TEXT = "Load a patient folder to generate the axial view and filtered DVHs."
DVH_NO_SELECTION_STATUS_TEXT = "Select structures in the DVH tab to generate DVHs."
DVH_NO_CURVES_STATUS_TEXT = "No structures produced DVH data on the current CT grid."


@dataclass(frozen=True)
class DvhReadoutState:
    dose_gy: float
    volume_pct: float
    volume_cc: float
    text: str


@dataclass(frozen=True)
class DvhPlotCurveSpec:
    normalized_name: str
    plot_dose_bins: np.ndarray
    plot_volume_pct: np.ndarray
    color_rgb: Tuple[int, int, int]
    width: int


@dataclass(frozen=True)
class DvhRefreshRequest:
    selected_names: List[str]
    reusable_mask_cache: object


@dataclass(frozen=True)
class DvhTaskCompletionState:
    selected_curve_name: Optional[str]
    status_text: Optional[str]
    should_clear_selection: bool


@dataclass(frozen=True)
class DvhVisibilityRefreshPlan:
    should_refresh_from_scratch: bool
    should_invalidate_jobs: bool
    should_clear_status: bool


def get_curve_for_name(curves: Sequence[DVHCurve], normalized_name: str) -> Optional[DVHCurve]:
    for curve in curves:
        if normalize_structure_name(curve.name) == normalized_name:
            return curve
    return None


def get_current_curve_names(curves: Sequence[DVHCurve]) -> List[str]:
    return [normalize_structure_name(curve.name) for curve in curves]


def get_dvh_missing_inputs_status_text() -> str:
    return DVH_MISSING_INPUTS_STATUS_TEXT


def get_dvh_no_selection_status_text() -> str:
    return DVH_NO_SELECTION_STATUS_TEXT


def get_dvh_no_curves_status_text() -> str:
    return DVH_NO_CURVES_STATUS_TEXT


def get_dvh_task_failed_status_text(error_message: str) -> str:
    return f"DVH computation failed: {error_message}"


def get_dvh_selection_prompt(curve_name: str) -> str:
    return f"Selected {curve_name}. Move the mouse over the DVH plot to inspect dose and volume."


def get_dvh_curve_highlight_width(normalized_name: str, selected_name: Optional[str]) -> int:
    return 4 if normalized_name == selected_name else 2


def get_visible_dvh_curves(
    curves: Sequence[DVHCurve],
    visibility_resolver: Callable[[str], bool],
) -> List[DVHCurve]:
    return [
        curve
        for curve in curves
        if visibility_resolver(normalize_structure_name(curve.name))
    ]


def get_selected_dvh_structure_names(
    rtstruct: Optional[RTStructData],
    visibility_resolver: Callable[[str], bool],
) -> List[str]:
    if rtstruct is None:
        return []

    return [
        normalize_structure_name(structure.name)
        for structure in rtstruct.structures
        if visibility_resolver(normalize_structure_name(structure.name))
    ]


def build_selected_dvh_rtstruct(
    rtstruct: Optional[RTStructData],
    selected_names: Sequence[str],
) -> Optional[RTStructData]:
    if rtstruct is None:
        return None

    selected_name_set = set(selected_names)
    selected_structures = [
        structure
        for structure in rtstruct.structures
        if normalize_structure_name(structure.name) in selected_name_set
    ]
    return RTStructData(
        structures=selected_structures,
        frame_of_reference_uid=rtstruct.frame_of_reference_uid,
    )


def resolve_selected_curve_name(
    curves: Sequence[DVHCurve],
    selected_name: Optional[str],
    visibility_resolver: Callable[[str], bool],
) -> Optional[str]:
    if selected_name is None:
        return None
    curve = get_curve_for_name(curves, selected_name)
    if curve is None or not visibility_resolver(selected_name):
        return None
    return selected_name


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


def build_dvh_plot_curve_specs(
    curves: Sequence[DVHCurve],
    visibility_resolver: Callable[[str], bool],
    selected_name: Optional[str],
) -> List[DvhPlotCurveSpec]:
    specs: List[DvhPlotCurveSpec] = []
    for curve in curves:
        normalized_name = normalize_structure_name(curve.name)
        if not visibility_resolver(normalized_name):
            continue
        plot_dose_bins, plot_volume_pct = get_dvh_plot_arrays(curve)
        specs.append(
            DvhPlotCurveSpec(
                normalized_name=normalized_name,
                plot_dose_bins=plot_dose_bins,
                plot_volume_pct=plot_volume_pct,
                color_rgb=curve.color_rgb,
                width=get_dvh_curve_highlight_width(normalized_name, selected_name),
            )
        )
    return specs


def build_dvh_refresh_request(
    selected_names: Sequence[str],
    selected_rtstruct: Optional[RTStructData],
    mask_cache: object,
    mask_cache_names: Sequence[str],
) -> Optional[DvhRefreshRequest]:
    if selected_rtstruct is None or not selected_rtstruct.structures:
        return None

    reusable_mask_cache = mask_cache
    if reusable_mask_cache is not None and list(mask_cache_names) != list(selected_names):
        reusable_mask_cache = None

    return DvhRefreshRequest(
        selected_names=list(selected_names),
        reusable_mask_cache=reusable_mask_cache,
    )


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


def compute_visible_structure_goal_evaluations(
    curves: Sequence[DVHCurve],
    structure_goals_by_name: Dict[str, List[StructureGoal]],
    selected_names: Sequence[str],
    precomputed: Optional[Dict[str, List[StructureGoalEvaluation]]] = None,
) -> Dict[str, List[StructureGoalEvaluation]]:
    selected_name_set = set(selected_names)
    if not selected_name_set or not curves:
        return {}

    if precomputed is not None:
        visible_selected_names = {
            normalize_structure_name(curve.name)
            for curve in curves
            if normalize_structure_name(curve.name) in selected_name_set
        }
        result = {
            normalize_structure_name(name): evaluations
            for name, evaluations in precomputed.items()
            if normalize_structure_name(name) in visible_selected_names
        }
        missing_names = [
            name
            for name in visible_selected_names
            if name not in result
        ]
        if not missing_names:
            return result
        computed = evaluate_visible_structure_goals(
            list(curves),
            structure_goals_by_name,
            list(missing_names),
        )
        result.update(computed)
        return result

    return evaluate_visible_structure_goals(
        list(curves),
        structure_goals_by_name,
        list(selected_name_set),
    )


def build_dvh_task_completion_state(
    curves: Sequence[DVHCurve],
    selected_name: Optional[str],
    visibility_resolver: Callable[[str], bool],
) -> DvhTaskCompletionState:
    resolved_selected_name = resolve_selected_curve_name(curves, selected_name, visibility_resolver)
    if curves:
        return DvhTaskCompletionState(
            selected_curve_name=resolved_selected_name,
            status_text=None,
            should_clear_selection=False,
        )
    return DvhTaskCompletionState(
        selected_curve_name=None,
        status_text=get_dvh_no_curves_status_text(),
        should_clear_selection=True,
    )


def can_reuse_current_dvh_curves(
    selected_names: Sequence[str],
    current_curve_names: Sequence[str],
    curves: Sequence[DVHCurve],
) -> bool:
    if not curves:
        return False
    current_curve_name_set = set(current_curve_names)
    return all(name in current_curve_name_set for name in selected_names)


def build_dvh_visibility_refresh_plan(
    selected_names: Sequence[str],
    current_curve_names: Sequence[str],
    curves: Sequence[DVHCurve],
) -> DvhVisibilityRefreshPlan:
    if not selected_names:
        return DvhVisibilityRefreshPlan(
            should_refresh_from_scratch=True,
            should_invalidate_jobs=False,
            should_clear_status=False,
        )

    if can_reuse_current_dvh_curves(selected_names, current_curve_names, curves):
        return DvhVisibilityRefreshPlan(
            should_refresh_from_scratch=False,
            should_invalidate_jobs=True,
            should_clear_status=True,
        )

    return DvhVisibilityRefreshPlan(
        should_refresh_from_scratch=True,
        should_invalidate_jobs=False,
        should_clear_status=False,
    )
