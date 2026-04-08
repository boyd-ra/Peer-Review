from __future__ import annotations

import gc
import html
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
import pyqtgraph.functions as pgfn
from pyqtgraph.exporters import ImageExporter
from PySide6 import QtCore, QtGui, QtPrintSupport, QtWidgets
try:
    from scipy.ndimage import binary_dilation
except Exception:  # pragma: no cover - optional runtime dependency
    binary_dilation = None

from peer_helpers import (
    build_structure_slice_mask,
    compute_image_view_bounds,
    compute_single_structure_high_accuracy_curve,
    dose_at_volume_cc,
    estimate_structure_geometry_metrics,
    fill_binary_holes_2d,
    get_dvh_method_signature,
    get_ptv_dose_levels_gy,
    normalize_structure_name,
    sample_dose_to_ct_slice,
    volume_cc_at_dose_gy,
    volume_pct_at_dose_gy,
)
from peer_cache import (
    callable_signature_hash,
    build_review_cache_payload,
    deserialize_dvh_curve as deserialize_dvh_curve_payload,
    deserialize_goal_evaluations as deserialize_goal_evaluations_payload,
    deserialize_structure_goals as deserialize_structure_goals_payload,
    deserialize_target_table_rows as deserialize_target_table_rows_payload,
    get_ct_geometry_signature as compute_ct_geometry_signature,
    get_derived_array_cache_path as compute_derived_array_cache_path,
    get_derived_array_cache_signature as compute_derived_array_cache_signature,
    get_derived_cache_structures as select_derived_cache_structures,
    get_dvh_cache_path as compute_dvh_cache_path,
    load_derived_array_cache as load_derived_array_cache_file,
    load_review_cache_file,
    prepare_review_cache_state,
    PreparedReviewCacheState,
    ReviewCacheFileData,
    save_derived_array_cache as save_derived_array_cache_file,
    serialize_dvh_curve as serialize_dvh_curve_payload,
    serialize_goal_evaluations as serialize_goal_evaluations_payload,
    serialize_structure_goals as serialize_structure_goals_payload,
    serialize_target_table_rows as serialize_target_table_rows_payload,
    write_json_atomic,
)
from peer_constraints_table import (
    build_constraint_editor_preview_state,
    build_initial_constraint_editor_state,
    build_constraints_table_presentation_rows,
    build_custom_constraint_from_editor as build_custom_constraint_from_editor_helper,
    compose_constraint_note_text as compose_constraint_note_text_helper,
    custom_constraint_exists as custom_constraint_exists_helper,
    get_computed_constraint_note_text as get_computed_constraint_note_text_helper,
    get_constraint_goal_key as get_constraint_goal_key_helper,
    get_constraint_evaluations_for_structure as get_constraint_evaluations_for_structure_helper,
    get_constraints_table_column_widths,
    get_min_bladder_volume_note_text as get_min_bladder_volume_note_text_helper,
    parse_constraint_goal_input as parse_constraint_goal_input_helper,
    prostate_constraint_summary_enabled as prostate_constraint_summary_enabled_helper,
    refresh_constraints_table,
)
from peer_dvh_controller import (
    build_dvh_readout_state,
    build_dvh_plot_curve_specs,
    build_dvh_refresh_request,
    build_dvh_task_completion_state,
    build_dvh_visibility_refresh_plan,
    build_selected_dvh_rtstruct as build_selected_dvh_rtstruct_helper,
    compute_visible_structure_goal_evaluations,
    find_nearest_dvh_curve_name as find_nearest_dvh_curve_name_helper,
    get_curve_for_name as get_curve_for_name_helper,
    get_current_curve_names as get_current_dvh_curve_names_helper,
    get_dvh_curve_highlight_width,
    get_dvh_missing_inputs_status_text,
    get_dvh_no_selection_status_text,
    get_dvh_plot_arrays as get_dvh_plot_arrays_helper,
    get_dvh_selection_prompt,
    get_dvh_task_failed_status_text,
    get_selected_dvh_structure_names as get_selected_dvh_structure_names_helper,
    get_visible_dvh_curves as get_visible_dvh_curves_helper,
    get_visible_dvh_view_range,
    resolve_selected_curve_name,
)
from peer_io import (
    get_constraints_workbook_path,
    list_constraints_workbook_sheets,
    load_combined_rtdose,
    load_rtdose,
    load_rtstruct,
    load_structure_constraints_sheet,
)
from peer_loader import (
    build_axial_cine_plan,
    build_load_timing_report_text,
    get_review_cache_availability,
    PatientActivationPreparationManager,
    PatientActivationPreparationPayload,
    PatientPreloadManager,
    PatientPreloadPayload,
    PrecomputedPatientViewState,
    prepare_patient_preload_payload,
    ReviewCacheAvailability,
)
from peer_rendering import (
    apply_isodose_items,
    apply_polyline_specs,
    build_active_isodose_levels,
    build_axial_hover_text,
    build_axial_overlay_positions,
    build_axial_render_state,
    build_max_dose_center_points,
    build_max_dose_marker_state,
    build_orthogonal_render_state,
    clear_overlay_items as clear_overlay_items_helper,
    resolve_axial_indices,
)
from peer_models import (
    CTVolume,
    DVHCurve,
    DoseVolume,
    ImageViewBounds,
    RTPlanPhase,
    RTStructData,
    StructureGoal,
    StructureGoalEvaluation,
    StructureSliceContours,
)
from peer_targets import (
    build_target_table_rows as build_target_table_rows_helper,
    build_target_notes_for_save as build_target_notes_for_save_helper,
    compute_stereotactic_indices as compute_stereotactic_indices_helper,
    compute_stereotactic_owned_volume_cc as compute_stereotactic_owned_volume_cc_helper,
    compose_target_note_text as compose_target_note_text_helper,
    extract_manual_target_notes as extract_manual_target_notes_helper,
    get_default_stereotactic_dose_text as get_default_stereotactic_dose_text_helper,
    get_phase_target_assignments as get_phase_target_assignments_helper,
    get_preferred_manual_target_parent_name as get_preferred_manual_target_parent_name_helper,
    get_primary_target_context as get_primary_target_context_helper,
    get_sorted_ptv_structures as get_sorted_ptv_structures_helper,
    get_stereotactic_competing_ptv_entries as get_stereotactic_competing_ptv_entries_helper,
    get_target_fraction_count as get_target_fraction_count_helper,
    get_target_row_reference_dose_text as get_target_row_reference_dose_text_helper,
    localize_stereotactic_extra_mask as localize_stereotactic_extra_mask_helper,
    resolve_nested_target_names,
    stereotactic_summary_enabled as stereotactic_summary_enabled_helper,
    target_table_rows_require_recompute as target_table_rows_require_recompute_helper,
)
from peer_targets_table import (
    build_target_table_presentation_rows,
    create_target_coverage_cell_widget as create_target_coverage_cell_widget_helper,
    create_target_name_cell_widget as create_target_name_cell_widget_helper,
    create_target_note_button_widget as create_target_note_button_widget_helper,
    get_target_table_column_widths,
    populate_target_table_rows,
)
from peer_widgets import LineSwatchWidget, RangeSlider, WindowLevelSlider
from peer_viewer_support import (
    DVHComputationManager,
    StructureListManager,
    build_file_fingerprint,
    build_file_fingerprints,
    file_fingerprint_list_matches,
    file_fingerprint_matches,
)


NO_CONSTRAINTS_SHEET_LABEL = "------"
MAX_TISSUE_ROW_LABEL = "Max Tissue"
MAX_TISSUE_ROW_NAME = normalize_structure_name(MAX_TISSUE_ROW_LABEL)
logger = logging.getLogger(__name__)


class RectZoomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("enableMenu", False)
        super().__init__(*args, **kwargs)
        self.setMouseMode(self.RectMode)

    def mouseDragEvent(self, event, axis=None):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            event.ignore()
            return
        self.setMouseMode(self.RectMode)
        super().mouseDragEvent(event, axis=axis)

    def wheelEvent(self, event, axis=None):
        event.ignore()


def build_app_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#2f2f2f"))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#f2f2f2"))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#232323"))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#2b2b2b"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#232323"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#f2f2f2"))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#f2f2f2"))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#4a4a4a"))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#f2f2f2"))
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#3e7dd8"))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.Light, QtGui.QColor("#5d5d5d"))
    palette.setColor(QtGui.QPalette.ColorRole.Midlight, QtGui.QColor("#505050"))
    palette.setColor(QtGui.QPalette.ColorRole.Dark, QtGui.QColor("#1b1b1b"))
    palette.setColor(QtGui.QPalette.ColorRole.Mid, QtGui.QColor("#363636"))
    palette.setColor(QtGui.QPalette.ColorRole.Shadow, QtGui.QColor("#101010"))
    palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text, QtGui.QColor("#8f8f8f"))
    palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#8f8f8f"))
    palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#9b9b9b"))
    return palette


def build_app_stylesheet() -> str:
    return """
QMainWindow, QWidget {
  background-color: #2f2f2f;
  color: #f2f2f2;
}
QToolBar {
  background-color: #343434;
  border: none;
  spacing: 6px;
  padding: 4px 6px;
}
QToolBar::separator {
  background: #505050;
  width: 1px;
  margin: 6px 8px;
}
QToolButton, QPushButton {
  background-color: #4a4a4a;
  color: #f2f2f2;
  border: 1px solid #616161;
  border-radius: 5px;
  padding: 4px 10px;
}
QToolButton:hover, QPushButton:hover {
  background-color: #585858;
}
QToolButton:pressed, QPushButton:pressed {
  background-color: #666666;
}
QToolButton:disabled, QPushButton:disabled {
  background-color: #3a3a3a;
  color: #8f8f8f;
  border-color: #4a4a4a;
}
QLineEdit, QComboBox, QAbstractSpinBox {
  background-color: #232323;
  color: #f2f2f2;
  border: 1px solid #5c5c5c;
  border-radius: 4px;
  padding: 3px 6px;
  selection-background-color: #3e7dd8;
}
QComboBox::drop-down {
  border: none;
  width: 18px;
}
QComboBox QAbstractItemView {
  background-color: #232323;
  color: #f2f2f2;
  selection-background-color: #3e7dd8;
  selection-color: #ffffff;
}
QTabWidget::pane {
  border: 1px solid #4f4f4f;
  background-color: #2f2f2f;
}
QTabBar::tab {
  background-color: #444444;
  color: #f2f2f2;
  border: 1px solid #5f5f5f;
  border-bottom: none;
  padding: 6px 14px;
  min-width: 72px;
}
QTabBar::tab:selected {
  background-color: #2f2f2f;
}
QTabBar::tab:!selected:hover {
  background-color: #505050;
}
QHeaderView::section {
  background-color: #414141;
  color: #f2f2f2;
  border: 1px solid #555555;
  padding: 4px 6px;
}
QTableWidget, QListWidget {
  background-color: #232323;
  color: #f2f2f2;
  border: 1px solid #4f4f4f;
  alternate-background-color: #2b2b2b;
}
QTableWidget::item {
  color: #f2f2f2;
}
QTableWidget::item:selected, QListWidget::item:selected {
  background-color: #3e7dd8;
  color: #ffffff;
}
QCheckBox {
  color: #f2f2f2;
}
QCheckBox::indicator {
  width: 14px;
  height: 14px;
  border: 1px solid #7a7a7a;
  border-radius: 3px;
  background-color: #1c1c1c;
}
QCheckBox::indicator:checked {
  border: 1px solid #79e08f;
  background-color: #2f9e44;
}
QCheckBox::indicator:hover {
  border: 1px solid #a0a0a0;
}
QSlider::groove:horizontal {
  border: 1px solid #444444;
  height: 6px;
  background: #232323;
  border-radius: 3px;
}
QSlider::handle:horizontal {
  background: #d0d0d0;
  border: 1px solid #7a7a7a;
  width: 14px;
  margin: -5px 0;
  border-radius: 7px;
}
QStatusBar {
  background-color: #343434;
  color: #d0d0d0;
}
QScrollBar:vertical, QScrollBar:horizontal {
  background: #2a2a2a;
  border: none;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
  background: #5c5c5c;
  border-radius: 4px;
}
"""


def apply_app_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    app.setPalette(build_app_palette())
    app.setStyleSheet(build_app_stylesheet())


class RTPlanReviewWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radiotherapy Plan Review - powered by CommonSense™")
        self.resize(1450, 900)

        self.ct: Optional[CTVolume] = None
        self.dose: Optional[DoseVolume] = None
        self.rtstruct: Optional[RTStructData] = None
        self.rtstruct_path: Optional[str] = None
        self.current_patient_folder: Optional[str] = None
        self.patient_list_root_folder: Optional[str] = None
        self.patient_list_folders: List[str] = []
        self.current_patient_list_index: Optional[int] = None
        self.pending_patient_preload_index: Optional[int] = None
        self.pending_patient_preload_request_id: Optional[int] = None
        self.preloaded_patient_index: Optional[int] = None
        self.preloaded_patient_payload: Optional[PatientPreloadPayload] = None
        self.preloaded_patient_error: Optional[str] = None
        self.deferred_patient_view_refresh_token = 0
        self.patient_activation_ui_locked = False
        self.patient_activation_token = 0
        self.pending_patient_activation_prepare_request_id: Optional[int] = None
        self.pending_patient_activation_prepare_token: Optional[int] = None
        self.pending_patient_activation_prepare_result: Optional[PatientActivationPreparationPayload] = None
        self.pending_patient_activation_prepare_error: Optional[str] = None
        self.structure_filter_csv_path: Optional[str] = None
        self.constraints_sheet_name: Optional[str] = None
        self.available_constraint_sheet_names: List[str] = []
        self.structure_csv_order: List[str] = []
        self.csv_structure_goals_by_name: Dict[str, List[StructureGoal]] = {}
        self.custom_structure_goals_by_name: Dict[str, List[StructureGoal]] = {}
        self.structure_goals_by_name: Dict[str, List[StructureGoal]] = {}
        self.structure_goal_evaluations: Dict[str, List[StructureGoalEvaluation]] = {}
        self.plan_phases: List[RTPlanPhase] = []
        self.current_rtplan_paths: List[str] = []
        self.constraint_notes: Dict[str, str] = {}
        self.target_notes: Dict[str, str] = {}
        self.patient_plan_lines: Optional[Tuple[str, ...]] = None
        self.displayed_dose_plane: Optional[np.ndarray] = None
        self.sampled_dose_volume_ct: Optional[np.ndarray] = None
        self.structure_mask_cache: Optional[List[Dict[int, np.ndarray]]] = None
        self.structure_mask_cache_names: List[str] = []
        self.image_view_bounds: Optional[ImageViewBounds] = None
        self.dvh_curves: List[DVHCurve] = []
        self.dvh_plot_items: Dict[str, pg.PlotDataItem] = {}
        self.selected_dvh_curve_name: Optional[str] = None
        self.dvh_request_structure_names: Dict[int, List[str]] = {}
        self.dvh_structure_volume_cache: Dict[str, float] = {}
        self.dvh_ptv_coverage_cache: Dict[str, str] = {}
        self.dvh_structure_goal_evaluation_cache: Dict[str, List[StructureGoalEvaluation]] = {}
        self.latest_timing_entries: List[Tuple[str, Optional[float]]] = []
        self.latest_timing_folder: Optional[str] = None
        self.latest_timing_csv_path: Optional[str] = None
        self.latest_timing_rtstruct_path: Optional[str] = None
        self.latest_timing_rtdose_paths: List[str] = []
        self.constraint_workbook_error: Optional[str] = None
        self.phase_dose_volumes_by_path: Dict[str, DoseVolume] = {}
        self.phase_dose_plane_cache: Dict[Tuple[str, int], np.ndarray] = {}
        self.phase_dose_volume_ct_cache: Dict[str, np.ndarray] = {}
        self.target_curve_cache: Dict[Tuple[str, str], Optional[DVHCurve]] = {}
        self.target_metrics_cache: Dict[Tuple[str, str, float], Tuple[float, float, float]] = {}
        self.stereotactic_metrics_cache: Dict[Tuple[str, str, float, int], Tuple[float, float, float, float, float]] = {}
        self.stereotactic_volume_context_cache: Dict[Tuple[str, str, float], Optional[Dict[str, object]]] = {}
        self.max_tissue_dose_gy_cache: Optional[float] = None
        self.max_tissue_index_zyx: Optional[Tuple[int, int, int]] = None
        self.ptv_union_slice_mask_cache: Optional[Dict[int, np.ndarray]] = None
        self.ptv_union_volume_mask_cache: Optional[np.ndarray] = None
        self.target_slice_mask_cache: Dict[str, Dict[int, np.ndarray]] = {}
        self.target_containment_cache: Dict[str, List[str]] = {}
        self.structure_volume_mask_cache: Dict[str, np.ndarray] = {}
        self.structure_geometry_volume_cache: Dict[str, float] = {}
        self.cached_target_table_rows: Optional[List[Dict[str, object]]] = None
        self.defer_sidebar_summary_metrics: bool = False
        self.restore_saved_results_without_calculation: bool = False
        self.stereotactic_target_dose_text_by_name: Dict[str, str] = {}
        self.constraints_table_refresh_pending = False
        self.targets_table_refresh_pending = False
        self.dvh_plot_refresh_pending = False
        self.constraint_editor_state: Optional[Dict[str, str]] = None
        self.constraint_editor_widgets: Dict[str, QtWidgets.QWidget] = {}
        self.hidden_structure_names: set[str] = set()
        self.additional_target_subvolume_names: set[str] = set()
        self.pending_saved_dvh_selected_names: Optional[List[str]] = None
        self.structure_filter_dialog: Optional[QtWidgets.QDialog] = None
        self.structure_filter_tree_widget: Optional[QtWidgets.QTreeWidget] = None
        self.current_row: int = 0
        self.current_col: int = 0
        self.max_dose_index_zyx: Optional[Tuple[int, int, int]] = None

        self.axial_contour_items: List[pg.PlotCurveItem] = []
        self.sagittal_contour_items: List[pg.PlotCurveItem] = []
        self.coronal_contour_items: List[pg.PlotCurveItem] = []
        self.axial_isodose_items: List[pg.IsocurveItem] = []
        self.sagittal_isodose_items: List[pg.IsocurveItem] = []
        self.coronal_isodose_items: List[pg.IsocurveItem] = []
        self.isodose_colors: List[Tuple[int, int, int]] = [
            (0, 0, 0),
            (255, 215, 0),
            (0, 255, 255),
            (50, 205, 50),
            (255, 140, 0),
        ]
        self.autoscroll_default_speed_mm_per_s = 7.5
        self.autoscroll_speed_step_mm_per_s = 2.5
        self.autoscroll_interval_min_ms = 30
        self.autoscroll_interval_max_ms = 400
        self.autoscroll_timer = QtCore.QTimer(self)
        self.autoscroll_timer.setInterval(120)
        self.autoscroll_timer.setSingleShot(True)
        self.autoscroll_direction = 1
        self.patient_transition_overlay_process: Optional[subprocess.Popen] = None
        self.patient_transition_overlay_temp_dir: Optional[str] = None
        self.isodose_refresh_timer = QtCore.QTimer(self)
        self.isodose_refresh_timer.setSingleShot(True)
        self.isodose_refresh_timer.timeout.connect(self.refresh_all_views)

        self._build_ui()
        self.axial_structure_list = StructureListManager([("axial", self.structures_list)])
        self.dvh_structure_list = StructureListManager([("dvh", self.dvh_structures_list)])
        self.dvh_job_manager = DVHComputationManager()
        self.patient_preload_manager = PatientPreloadManager()
        self.patient_activation_prepare_manager = PatientActivationPreparationManager()
        self._connect_signals()
        self.refresh_constraint_sheet_combo()
        self.update_dose_range_controls()

    def _build_ui(self):
        self._create_actions()
        self._create_toolbar()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        self.progress_status_label = QtWidgets.QLabel("")
        self.progress_status_label.setStyleSheet("color: #d0d0d0; padding-left: 4px;")
        self.statusBar().addWidget(self.progress_status_label, 1)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::tab-bar { alignment: right; }")
        self.tabs.tabBar().setExpanding(False)
        root.addWidget(self.tabs)

        constraints_tab = QtWidgets.QWidget()
        self.constraints_tab = constraints_tab
        constraints_layout = QtWidgets.QVBoxLayout(constraints_tab)
        constraints_top_row = QtWidgets.QHBoxLayout()
        self.add_constraint_button = QtWidgets.QPushButton("Add Constraint")
        self.add_constraint_button.setFixedWidth(128)
        self.add_constraint_button.setEnabled(False)
        constraints_top_row.addWidget(self.add_constraint_button)
        self.constraint_sheet_combo = QtWidgets.QComboBox()
        self.constraint_sheet_combo.setMinimumWidth(180)
        self.constraint_sheet_combo.setEnabled(False)
        constraints_top_row.addWidget(self.constraint_sheet_combo)
        constraints_top_row.addStretch(1)
        constraints_layout.addLayout(constraints_top_row)
        self.constraints_table = QtWidgets.QTableWidget(0, 6)
        self.constraints_table.setHorizontalHeaderLabels(["OAR", "Metric", "Goal", "Result", "Notes", "Note"])
        self.constraints_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.constraints_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.constraints_table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.constraints_table.setShowGrid(False)
        self.constraints_table.setAlternatingRowColors(False)
        self.constraints_table.setWordWrap(False)
        self.constraints_table.verticalHeader().hide()
        self.constraints_table.horizontalHeader().setStretchLastSection(False)
        self.constraints_table.horizontalHeader().setHighlightSections(False)
        self.constraints_table.horizontalHeader().setDefaultAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        for column_index in range(6):
            self.constraints_table.horizontalHeader().setSectionResizeMode(
                column_index,
                QtWidgets.QHeaderView.ResizeMode.Fixed,
            )
        table_font = self.constraints_table.font()
        table_font.setPointSize(11)
        self.constraints_table.setFont(table_font)
        header_font = self.constraints_table.horizontalHeader().font()
        header_font.setPointSize(11)
        header_font.setBold(True)
        self.constraints_table.horizontalHeader().setFont(header_font)
        self.constraints_table.verticalHeader().setDefaultSectionSize(30)
        constraints_layout.addWidget(self.constraints_table, 1)

        self.targets_tab = QtWidgets.QWidget()
        targets_layout = QtWidgets.QVBoxLayout(self.targets_tab)
        self.targets_table = QtWidgets.QTableWidget(0, 6)
        self.targets_table.setHorizontalHeaderLabels(["PTV", "Coverage @ Rx", "Min Dose", "Max Dose", "Notes", "Note"])
        self.targets_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.targets_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.targets_table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.targets_table.setShowGrid(False)
        self.targets_table.setAlternatingRowColors(False)
        self.targets_table.setWordWrap(True)
        self.targets_table.verticalHeader().hide()
        self.targets_table.horizontalHeader().setStretchLastSection(False)
        self.targets_table.horizontalHeader().setHighlightSections(False)
        self.targets_table.horizontalHeader().setDefaultAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        for column_index in range(6):
            self.targets_table.horizontalHeader().setSectionResizeMode(
                column_index,
                QtWidgets.QHeaderView.ResizeMode.Fixed,
            )
        targets_table_font = self.targets_table.font()
        targets_table_font.setPointSize(11)
        self.targets_table.setFont(targets_table_font)
        targets_header_font = self.targets_table.horizontalHeader().font()
        targets_header_font.setPointSize(11)
        targets_header_font.setBold(True)
        self.targets_table.horizontalHeader().setFont(targets_header_font)
        self.targets_table.verticalHeader().setDefaultSectionSize(30)
        targets_layout.addWidget(self.targets_table, 1)

        axial_tab = QtWidgets.QWidget()
        self.axial_tab_widget = axial_tab
        axial_layout = QtWidgets.QHBoxLayout(axial_tab)

        sidebar_widget = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_widget)
        sidebar_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.slice_label = QtWidgets.QLabel("Slice: -/-")
        self.z_label = QtWidgets.QLabel("z: -")
        self.window_label = QtWidgets.QLabel("WL/WW: 40 / 400")

        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_prev_button = QtWidgets.QPushButton("-")
        self.slice_next_button = QtWidgets.QPushButton("+")
        for button, tooltip in (
            (self.slice_prev_button, "Previous slice"),
            (self.slice_next_button, "Next slice"),
        ):
            button.setFixedWidth(32)
            button.setToolTip(tooltip)

        self.window_level_slider = WindowLevelSlider()
        self.window_level_slider.setRange(-1200, 2000)
        self.window_level_slider.setWindowLevel(400, 40)
        self.reset_window_level_button = QtWidgets.QPushButton("Reset")

        self.dose_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.dose_opacity_slider.setRange(0, 100)
        self.dose_opacity_slider.setValue(75)
        self.dose_overlay_enabled = True
        self.dose_toggle_button = QtWidgets.QPushButton("On")
        self.dose_toggle_button.setCheckable(True)
        self.dose_toggle_button.setChecked(True)
        self.max_dose_button = QtWidgets.QPushButton("Max")

        self.dose_range_label = QtWidgets.QLabel("Dose range: 0.00 Gy - 100%")
        self.dose_min_edit = QtWidgets.QLineEdit("0.00")
        self.dose_min_edit.setFixedWidth(72)
        self.dose_max_edit = QtWidgets.QLineEdit("0.00")
        self.dose_max_edit.setFixedWidth(72)
        dose_validator = QtGui.QDoubleValidator(0.0, 100000.0, 2, self)
        dose_validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        self.dose_min_edit.setValidator(dose_validator)
        self.dose_max_edit.setValidator(QtGui.QDoubleValidator(0.0, 100000.0, 2, self))
        self.dose_range_slider = RangeSlider()
        self.dose_range_slider.setRange(0, 1000)
        self.dose_range_slider.setValues(0, 1000)

        dose_range_widget = QtWidgets.QWidget()
        dose_range_layout = QtWidgets.QHBoxLayout(dose_range_widget)
        dose_range_layout.setContentsMargins(0, 0, 0, 0)
        dose_range_layout.setSpacing(8)
        dose_range_layout.addWidget(self.dose_min_edit)
        dose_range_layout.addWidget(self.dose_range_slider, 1)
        dose_range_layout.addWidget(self.dose_max_edit)

        self.autoscroll_button = QtWidgets.QPushButton("Autoscroll")
        self.autoscroll_button.setCheckable(True)
        self.autoscroll_slower_button = QtWidgets.QPushButton("-")
        self.autoscroll_faster_button = QtWidgets.QPushButton("+")
        self.autoscroll_button.setFixedWidth(88)
        self.autoscroll_button.setFixedHeight(24)
        autoscroll_button_style = (
            "QPushButton { background-color: rgba(120, 120, 120, 235); color: white; "
            "border: 1px solid rgba(210, 210, 210, 120); border-radius: 4px; "
            "padding: 1px 6px; font-size: 11px; }"
            "QPushButton:pressed { background-color: rgba(145, 145, 145, 235); }"
            "QPushButton:checked { background-color: rgba(165, 165, 165, 235); color: black; }"
        )
        self.autoscroll_button.setStyleSheet(autoscroll_button_style)
        for button, tooltip in (
            (self.autoscroll_slower_button, "Slow down auto scroll"),
            (self.autoscroll_faster_button, "Speed up auto scroll"),
        ):
            button.setFixedWidth(24)
            button.setFixedHeight(24)
            button.setToolTip(tooltip)
            button.setStyleSheet(autoscroll_button_style)

        self.structures_list = QtWidgets.QListWidget()
        self.structures_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.structures_list.setSpacing(1)
        self.patient_name_label = QtWidgets.QLabel("")
        self.patient_name_label.setWordWrap(False)
        self.patient_name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.patient_name_label.setMargin(2)
        patient_name_font = self.patient_name_label.font()
        patient_name_font.setPointSize(patient_name_font.pointSize() + 1)
        patient_name_font.setBold(True)
        self.patient_name_label.setFont(patient_name_font)
        self.patient_name_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.patient_name_label.setVisible(False)
        self.patient_plan_label = QtWidgets.QLabel("")
        self.patient_plan_label.setWordWrap(False)
        self.patient_plan_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.patient_plan_label.setMargin(2)
        self.patient_plan_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.patient_plan_label.setVisible(False)

        self.patient_summary_widget = QtWidgets.QWidget()
        patient_summary_layout = QtWidgets.QHBoxLayout(self.patient_summary_widget)
        patient_summary_layout.setContentsMargins(6, 2, 12, 0)
        patient_summary_layout.setSpacing(10)
        patient_summary_layout.addWidget(self.patient_name_label)
        patient_summary_layout.addWidget(self.patient_plan_label)
        self.patient_summary_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.tabs.setCornerWidget(self.patient_summary_widget, QtCore.Qt.Corner.TopLeftCorner)

        self.axial_sidebar_widget = sidebar_widget
        sidebar_layout.addWidget(QtWidgets.QLabel("Structures"))
        sidebar_layout.addWidget(self.structures_list, 1)
        sidebar_widget.setMinimumWidth(280)
        axial_layout.addWidget(sidebar_widget, 0)

        viewer_widget = QtWidgets.QWidget()
        self.viewer_widget = viewer_widget
        viewer_layout = QtWidgets.QVBoxLayout(viewer_widget)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(12)

        viewer_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        sagittal_widget = pg.GraphicsLayoutWidget()
        self.sagittal_graphics_widget = sagittal_widget
        self.sagittal_view = sagittal_widget.addViewBox(lockAspect=True, invertY=True)
        self.sagittal_view.setMenuEnabled(False)
        self.sagittal_view.enableAutoRange()
        self.sagittal_ct_item = pg.ImageItem(axisOrder="row-major")
        self.sagittal_dose_item = pg.ImageItem(axisOrder="row-major")
        self.sagittal_max_marker = pg.ScatterPlotItem(symbol="x", size=14, pen=pg.mkPen((0, 255, 255), width=2))
        self.sagittal_ct_item.setZValue(0)
        self.sagittal_dose_item.setZValue(1)
        self.sagittal_max_marker.setZValue(3)
        self.sagittal_view.addItem(self.sagittal_ct_item)
        self.sagittal_view.addItem(self.sagittal_dose_item)
        self.sagittal_view.addItem(self.sagittal_max_marker)

        coronal_widget = pg.GraphicsLayoutWidget()
        self.coronal_graphics_widget = coronal_widget
        self.coronal_view = coronal_widget.addViewBox(lockAspect=True, invertY=True)
        self.coronal_view.setMenuEnabled(False)
        self.coronal_view.enableAutoRange()
        self.coronal_ct_item = pg.ImageItem(axisOrder="row-major")
        self.coronal_dose_item = pg.ImageItem(axisOrder="row-major")
        self.coronal_max_marker = pg.ScatterPlotItem(symbol="x", size=14, pen=pg.mkPen((0, 255, 255), width=2))
        self.coronal_ct_item.setZValue(0)
        self.coronal_dose_item.setZValue(1)
        self.coronal_max_marker.setZValue(3)
        self.coronal_view.addItem(self.coronal_ct_item)
        self.coronal_view.addItem(self.coronal_dose_item)
        self.coronal_view.addItem(self.coronal_max_marker)

        axial_widget = pg.GraphicsLayoutWidget()
        self.axial_graphics_widget = axial_widget
        self.axial_view = axial_widget.addViewBox(lockAspect=True, invertY=True)
        self.axial_view.setMenuEnabled(False)
        self.axial_view.enableAutoRange()

        self.ct_item = pg.ImageItem(axisOrder="row-major")
        self.dose_item = pg.ImageItem(axisOrder="row-major")
        self.axial_max_marker = pg.ScatterPlotItem(symbol="x", size=14, pen=pg.mkPen((0, 255, 255), width=2))
        self.ct_item.setZValue(0)
        self.dose_item.setZValue(1)
        self.axial_max_marker.setZValue(3)
        self.axial_view.addItem(self.ct_item)
        self.axial_view.addItem(self.dose_item)
        self.axial_view.addItem(self.axial_max_marker)

        self.crosshair_text = pg.TextItem(anchor=(1, 0))
        self.axial_view.addItem(self.crosshair_text)
        self.crosshair_text.hide()
        self.axial_readout_label = QtWidgets.QLabel(axial_widget)
        self.axial_readout_label.setStyleSheet(
            "QLabel { background-color: rgba(0, 0, 0, 210); color: white; "
            "padding: 3px 7px; border-radius: 4px; font-size: 11px; }"
        )
        self.axial_readout_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop
        )
        self.axial_readout_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.axial_readout_label.hide()
        self.axial_autoscroll_overlay = QtWidgets.QWidget(axial_widget)
        self.axial_autoscroll_overlay.setStyleSheet(
            "QWidget { background-color: rgba(0, 0, 0, 210); border-radius: 4px; }"
        )
        self.axial_autoscroll_overlay.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        autoscroll_overlay_layout = QtWidgets.QHBoxLayout(self.axial_autoscroll_overlay)
        autoscroll_overlay_layout.setContentsMargins(5, 3, 5, 3)
        autoscroll_overlay_layout.setSpacing(4)
        autoscroll_overlay_layout.addWidget(self.autoscroll_button)
        autoscroll_overlay_layout.addWidget(self.autoscroll_slower_button)
        autoscroll_overlay_layout.addWidget(self.autoscroll_faster_button)
        self.autoscroll_speed_label = QtWidgets.QLabel("-- mm/s")
        self.autoscroll_speed_label.setStyleSheet("color: white; font-size: 11px;")
        self.autoscroll_speed_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        autoscroll_overlay_layout.addWidget(self.autoscroll_speed_label)

        right_splitter.addWidget(sagittal_widget)
        right_splitter.addWidget(coronal_widget)
        right_splitter.setSizes([1, 1])

        viewer_splitter.addWidget(axial_widget)
        viewer_splitter.addWidget(right_splitter)
        viewer_splitter.setStretchFactor(0, 3)
        viewer_splitter.setStretchFactor(1, 1)

        viewer_layout.addWidget(viewer_splitter, 19)

        bottom_controls_widget = QtWidgets.QWidget()
        bottom_controls_layout = QtWidgets.QGridLayout(bottom_controls_widget)
        bottom_controls_layout.setContentsMargins(0, 0, 0, 0)
        bottom_controls_layout.setHorizontalSpacing(16)
        bottom_controls_layout.setVerticalSpacing(4)

        image_slice_label = QtWidgets.QLabel("Slice scroll")
        image_window_label = QtWidgets.QLabel("Window / Level")
        window_level_widget = QtWidgets.QWidget()
        window_level_layout = QtWidgets.QHBoxLayout(window_level_widget)
        window_level_layout.setContentsMargins(0, 0, 0, 0)
        window_level_layout.setSpacing(8)
        window_level_layout.addWidget(self.reset_window_level_button)
        window_level_layout.addWidget(self.window_level_slider, 1)

        isodose_controls_widget = QtWidgets.QWidget()
        isodose_controls_layout = QtWidgets.QGridLayout(isodose_controls_widget)
        isodose_controls_layout.setContentsMargins(0, 0, 0, 0)
        isodose_controls_layout.setHorizontalSpacing(6)
        isodose_controls_layout.setSpacing(2)
        isodose_controls_layout.addWidget(QtWidgets.QLabel("Isodose"), 0, 0)
        self.isodose_edit_widgets: List[QtWidgets.QLineEdit] = []
        self.isodose_swatch_widgets: List[LineSwatchWidget] = []
        isodose_validator = QtGui.QDoubleValidator(0.0, 100000.0, 2, self)
        isodose_validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        isodose_positions = [(1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
        for color_rgb, (row, col) in zip(self.isodose_colors, isodose_positions):
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            swatch = LineSwatchWidget(color_rgb)
            row_layout.addWidget(swatch)
            edit = QtWidgets.QLineEdit("")
            edit.setFixedWidth(56)
            edit.setPlaceholderText("--")
            edit.setValidator(QtGui.QDoubleValidator(0.0, 100000.0, 2, self))
            row_layout.addWidget(edit)
            isodose_controls_layout.addWidget(row_widget, row, col)
            self.isodose_swatch_widgets.append(swatch)
            self.isodose_edit_widgets.append(edit)
        isodose_controls_layout.setColumnStretch(2, 1)

        dose_opacity_widget = QtWidgets.QWidget()
        dose_opacity_layout = QtWidgets.QHBoxLayout(dose_opacity_widget)
        dose_opacity_layout.setContentsMargins(0, 0, 0, 0)
        dose_opacity_layout.setSpacing(8)
        dose_opacity_layout.addWidget(QtWidgets.QLabel("Dose opacity"))
        dose_opacity_layout.addWidget(self.dose_opacity_slider, 1)
        dose_opacity_layout.addWidget(self.dose_toggle_button)
        dose_opacity_layout.addWidget(self.max_dose_button)
        dose_range_label = QtWidgets.QLabel("Dose range")

        slice_scroll_widget = QtWidgets.QWidget()
        slice_scroll_layout = QtWidgets.QHBoxLayout(slice_scroll_widget)
        slice_scroll_layout.setContentsMargins(0, 0, 0, 0)
        slice_scroll_layout.setSpacing(6)
        slice_scroll_layout.addWidget(self.slice_prev_button)
        slice_scroll_layout.addWidget(self.slice_slider, 1)
        slice_scroll_layout.addWidget(self.slice_next_button)

        bottom_controls_layout.addWidget(image_slice_label, 0, 0)
        bottom_controls_layout.addWidget(slice_scroll_widget, 0, 1)
        bottom_controls_layout.addWidget(image_window_label, 1, 0)
        bottom_controls_layout.addWidget(window_level_widget, 1, 1)
        bottom_controls_layout.addWidget(isodose_controls_widget, 0, 2, 2, 1)
        bottom_controls_layout.addWidget(dose_opacity_widget, 0, 3, 1, 2)
        bottom_controls_layout.addWidget(dose_range_label, 1, 3)
        bottom_controls_layout.addWidget(dose_range_widget, 1, 4)
        bottom_controls_layout.setColumnStretch(1, 1)
        bottom_controls_layout.setColumnStretch(4, 1)
        viewer_layout.addWidget(bottom_controls_widget, 1)

        axial_layout.addWidget(viewer_widget, 1)

        dvh_tab = QtWidgets.QWidget()
        self.dvh_tab = dvh_tab
        dvh_layout = QtWidgets.QHBoxLayout(dvh_tab)

        dvh_sidebar_widget = QtWidgets.QWidget()
        dvh_sidebar_layout = QtWidgets.QVBoxLayout(dvh_sidebar_widget)
        dvh_sidebar_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        dvh_sidebar_header_widget = QtWidgets.QWidget()
        dvh_sidebar_header_layout = QtWidgets.QHBoxLayout(dvh_sidebar_header_widget)
        dvh_sidebar_header_layout.setContentsMargins(0, 0, 0, 0)
        dvh_sidebar_header_layout.setSpacing(8)
        dvh_sidebar_header_layout.addWidget(QtWidgets.QLabel("Structures"))
        dvh_sidebar_header_layout.addStretch(1)
        self.clear_dvh_structures_button = QtWidgets.QPushButton("Clear")
        self.clear_dvh_structures_button.setFixedWidth(56)
        dvh_sidebar_header_layout.addWidget(self.clear_dvh_structures_button)
        dvh_sidebar_layout.addWidget(dvh_sidebar_header_widget)
        self.dvh_structures_list = QtWidgets.QListWidget()
        self.dvh_structures_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.dvh_structures_list.setSpacing(1)
        dvh_sidebar_layout.addWidget(self.dvh_structures_list, 1)
        dvh_sidebar_widget.setMinimumWidth(280)
        dvh_layout.addWidget(dvh_sidebar_widget, 0)

        dvh_content_widget = QtWidgets.QWidget()
        dvh_content_layout = QtWidgets.QVBoxLayout(dvh_content_widget)
        self.dvh_status_label = QtWidgets.QLabel("Load a patient folder to generate the axial view and filtered DVHs.")
        self.dvh_readout_label = QtWidgets.QLabel("Click a DVH curve to inspect dose and volume.")
        self.dvh_plot = pg.PlotWidget(viewBox=RectZoomViewBox())
        self.dvh_plot.setBackground("w")
        self.dvh_plot.showGrid(x=True, y=True, alpha=0.25)
        self.dvh_plot.setLabel("bottom", "Dose", units="Gy")
        self.dvh_plot.setLabel("left", "Volume", units="%")
        self.dvh_plot.setTitle("Dose-Volume Histogram")
        self.dvh_plot.setMouseEnabled(x=True, y=True)
        self.dvh_plot.setMenuEnabled(False)
        self.dvh_crosshair_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((90, 90, 90), width=1))
        self.dvh_crosshair_hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((90, 90, 90), width=1))
        self.dvh_curve_marker = pg.ScatterPlotItem(size=9)
        self.dvh_crosshair_vline.setZValue(4)
        self.dvh_crosshair_hline.setZValue(4)
        self.dvh_curve_marker.setZValue(5)

        dvh_content_layout.addWidget(self.dvh_status_label)
        dvh_content_layout.addWidget(self.dvh_readout_label)
        dvh_content_layout.addWidget(self.dvh_plot, 1)
        dvh_layout.addWidget(dvh_content_widget, 1)

        self.tabs.addTab(axial_tab, "Axial View")
        self.tabs.addTab(constraints_tab, "Constraints")
        self.tabs.addTab(self.targets_tab, "Targets")
        self.tabs.addTab(dvh_tab, "DVH")
        self.tabs.setCurrentWidget(axial_tab)

        self.statusBar().clearMessage()

    def _create_actions(self):
        self.load_patient_action = QtGui.QAction("Load Patient Folder", self)
        self.load_patient_list_action = QtGui.QAction("Load Patient List Folder", self)
        self.reset_view_action = QtGui.QAction("Reset View", self)
        self.clear_patient_action = QtGui.QAction("Clear", self)
        self.save_cache_action = QtGui.QAction("Save", self)
        self.print_report_action = QtGui.QAction("Print", self)
        self.structure_filter_action = QtGui.QAction("Structures", self)
        self.save_cache_action.setEnabled(False)
        self.print_report_action.setEnabled(False)
        self.structure_filter_action.setEnabled(False)

    def _create_toolbar(self):
        tb = self.addToolBar("Main")
        self.main_toolbar = tb
        tb.addAction(self.load_patient_action)
        tb.addAction(self.load_patient_list_action)
        self.patient_list_combo = QtWidgets.QComboBox(tb)
        self.patient_list_combo.setMinimumWidth(240)
        self.patient_list_combo.setEnabled(False)
        self.patient_list_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        tb.addWidget(self.patient_list_combo)
        self.next_patient_button = QtWidgets.QPushButton("Next", tb)
        self.next_patient_button.setEnabled(False)
        tb.addWidget(self.next_patient_button)
        tb.addSeparator()
        tb.addAction(self.reset_view_action)
        tb.addAction(self.clear_patient_action)
        tb.addAction(self.save_cache_action)
        tb.addAction(self.print_report_action)
        spacer = QtWidgets.QWidget(tb)
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        tb.addWidget(spacer)
        tb.addAction(self.structure_filter_action)

    def _connect_signals(self):
        self.load_patient_action.triggered.connect(self.on_load_patient_folder)
        self.load_patient_list_action.triggered.connect(self.on_load_patient_list_folder)
        self.reset_view_action.triggered.connect(self.on_reset_view)
        self.clear_patient_action.triggered.connect(self.on_clear_patient_session)
        self.save_cache_action.triggered.connect(self.on_save_dvh_cache)
        self.print_report_action.triggered.connect(self.on_print_report)
        self.structure_filter_action.triggered.connect(self.on_show_structure_filter_popup)
        self.patient_list_combo.activated.connect(self.on_patient_list_combo_activated)
        self.next_patient_button.clicked.connect(self.on_next_patient_clicked)
        self.reset_window_level_button.clicked.connect(self.on_reset_window_level)
        self.clear_dvh_structures_button.clicked.connect(self.on_clear_dvh_structures_clicked)
        self.max_dose_button.clicked.connect(self.on_go_to_max_dose)
        self.add_constraint_button.clicked.connect(self.on_add_constraint_clicked)
        self.constraint_sheet_combo.currentTextChanged.connect(self.on_constraint_sheet_changed)
        self.autoscroll_slower_button.clicked.connect(self.on_autoscroll_slower)
        self.autoscroll_faster_button.clicked.connect(self.on_autoscroll_faster)

        self.autoscroll_button.pressed.connect(self.on_autoscroll_button_pressed)
        self.autoscroll_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space), self)
        self.autoscroll_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.autoscroll_shortcut.activated.connect(self.toggle_autoscroll_shortcut)
        self.clear_dvh_curve_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self)
        self.clear_dvh_curve_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.clear_dvh_curve_shortcut.activated.connect(self.on_clear_dvh_curve_shortcut)
        self.slice_prev_button.clicked.connect(self.on_previous_slice)
        self.slice_next_button.clicked.connect(self.on_next_slice)
        self.slice_slider.valueChanged.connect(self.update_display)
        self.window_level_slider.valuesChanged.connect(self.refresh_all_views)
        self.dose_opacity_slider.valueChanged.connect(self.refresh_all_views)
        self.dose_toggle_button.toggled.connect(self.on_toggle_dose_overlay)
        self.dose_range_slider.valuesChanged.connect(self.on_dose_range_slider_changed)
        self.dose_min_edit.editingFinished.connect(self.on_dose_editing_finished)
        self.dose_max_edit.editingFinished.connect(self.on_dose_editing_finished)
        for edit in self.isodose_edit_widgets:
            edit.editingFinished.connect(self.on_isodose_editing_finished)
            edit.returnPressed.connect(self.on_isodose_editing_finished)
            edit.textChanged.connect(self.on_isodose_text_changed)
        self.autoscroll_button.toggled.connect(self.on_toggle_autoscroll)
        self.autoscroll_timer.timeout.connect(self.advance_autoscroll)
        self.axial_structure_list.visibilityChanged.connect(self.on_structure_visibility_changed)
        self.dvh_structure_list.visibilityChanged.connect(self.on_dvh_structure_visibility_changed)
        self.dvh_job_manager.finished.connect(self.on_dvh_task_finished)
        self.dvh_job_manager.failed.connect(self.on_dvh_task_failed)
        self.patient_preload_manager.finished.connect(self.on_patient_preload_finished)
        self.patient_preload_manager.failed.connect(self.on_patient_preload_failed)
        self.patient_activation_prepare_manager.finished.connect(self.on_patient_activation_prepare_finished)
        self.patient_activation_prepare_manager.failed.connect(self.on_patient_activation_prepare_failed)
        self.tabs.currentChanged.connect(self.axial_structure_list.refresh_layout)
        self.tabs.currentChanged.connect(self.dvh_structure_list.refresh_layout)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.axial_view.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.axial_view.scene().sigMouseClicked.connect(self.on_axial_clicked)
        self.dvh_mouse_proxy = pg.SignalProxy(
            self.dvh_plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.on_dvh_mouse_moved,
        )
        self.dvh_plot.scene().sigMouseClicked.connect(self.on_dvh_plot_clicked)

    def clear_viewer_image_items(self):
        empty_plane = np.zeros((1, 1), dtype=np.uint8)
        empty_rgba = np.zeros((1, 1, 4), dtype=np.uint8)

        self.ct_item.setImage(empty_plane, autoLevels=False, levels=(0, 255))
        self.dose_item.setImage(empty_rgba, autoLevels=False)
        self.sagittal_ct_item.setImage(empty_plane, autoLevels=False, levels=(0, 255))
        self.sagittal_dose_item.setImage(empty_rgba, autoLevels=False)
        self.coronal_ct_item.setImage(empty_plane, autoLevels=False, levels=(0, 255))
        self.coronal_dose_item.setImage(empty_rgba, autoLevels=False)

        self.clear_overlay_items(self.axial_view, self.axial_contour_items)
        self.clear_overlay_items(self.sagittal_view, self.sagittal_contour_items)
        self.clear_overlay_items(self.coronal_view, self.coronal_contour_items)
        self.clear_overlay_items(self.axial_view, self.axial_isodose_items)
        self.clear_overlay_items(self.sagittal_view, self.sagittal_isodose_items)
        self.clear_overlay_items(self.coronal_view, self.coronal_isodose_items)

        self.axial_max_marker.setData([], [])
        self.sagittal_max_marker.setData([], [])
        self.coronal_max_marker.setData([], [])
        self.crosshair_text.hide()
        self.axial_readout_label.hide()
        self.update_axial_overlay_positions()

    def clear_patient_session_state(self, *, quick_swap: bool = False):
        self.cancel_autoscroll()
        self.stop_patient_transition_overlay_process()
        self.dvh_job_manager.cancel_all()
        self.patient_activation_prepare_manager.cancel_all()
        self.clear_progress_status()
        self.deferred_patient_view_refresh_token += 1
        self.patient_activation_token += 1
        self.pending_patient_activation_prepare_request_id = None
        self.pending_patient_activation_prepare_token = None
        self.pending_patient_activation_prepare_result = None
        self.pending_patient_activation_prepare_error = None

        self.current_patient_folder = None
        self.latest_timing_entries = []
        self.latest_timing_folder = None
        self.latest_timing_csv_path = None
        self.latest_timing_rtstruct_path = None
        self.latest_timing_rtdose_paths = []
        self.constraint_workbook_error = None

        self.ct = None
        self.dose = None
        self.rtstruct = None
        self.rtstruct_path = None
        self.structure_filter_csv_path = None
        self.constraints_sheet_name = None
        self.available_constraint_sheet_names = []
        self.structure_csv_order = []
        self.csv_structure_goals_by_name = {}
        self.custom_structure_goals_by_name = {}
        self.structure_goals_by_name = {}
        self.structure_goal_evaluations = {}
        self.plan_phases = []
        self.current_rtplan_paths = []
        self.constraint_notes = {}
        self.target_notes = {}
        self.stereotactic_target_dose_text_by_name = {}
        self.patient_plan_lines = None
        self.image_view_bounds = None
        self.displayed_dose_plane = None
        self.sampled_dose_volume_ct = None
        self.structure_mask_cache = None
        self.structure_mask_cache_names = []
        self.dvh_curves = []
        self.dvh_plot_items = {}
        self.selected_dvh_curve_name = None
        self.dvh_request_structure_names = {}
        self.dvh_structure_volume_cache = {}
        self.dvh_ptv_coverage_cache = {}
        self.dvh_structure_goal_evaluation_cache = {}
        self.phase_dose_volumes_by_path = {}
        self.phase_dose_plane_cache = {}
        self.phase_dose_volume_ct_cache = {}
        self.target_curve_cache = {}
        self.target_metrics_cache = {}
        self.stereotactic_metrics_cache = {}
        self.stereotactic_volume_context_cache = {}
        self.max_tissue_dose_gy_cache = None
        self.max_tissue_index_zyx = None
        self.ptv_union_slice_mask_cache = None
        self.ptv_union_volume_mask_cache = None
        self.target_slice_mask_cache = {}
        self.target_containment_cache = {}
        self.structure_volume_mask_cache = {}
        self.structure_geometry_volume_cache = {}
        self.cached_target_table_rows = None
        self.defer_sidebar_summary_metrics = False
        self.restore_saved_results_without_calculation = False
        self.constraints_table_refresh_pending = False
        self.targets_table_refresh_pending = False
        self.dvh_plot_refresh_pending = False
        self.constraint_editor_state = None
        self.constraint_editor_widgets = {}
        self.hidden_structure_names = set()
        self.additional_target_subvolume_names = set()
        self.pending_saved_dvh_selected_names = None
        if self.structure_filter_tree_widget is not None:
            self.structure_filter_tree_widget.clear()
        if self.structure_filter_dialog is not None:
            self.structure_filter_dialog.hide()
        self.max_dose_index_zyx = None
        self.current_row = 0
        self.current_col = 0

        if quick_swap:
            return

        self.slice_slider.setRange(0, 0)
        self.slice_slider.setValue(0)
        self.slice_label.setText("Slice: -/-")
        self.z_label.setText("z: -")
        self.window_label.setText("WL/WW: 40 / 400")

        self.tabs.setCurrentIndex(0)
        self.refresh_constraint_sheet_combo(preferred_sheet_name=NO_CONSTRAINTS_SHEET_LABEL)
        self.update_patient_plan_label()
        self.update_autoscroll_speed_label()
        self.populate_structures_list()
        self.populate_isodose_controls()
        self.render_dvh_plot()
        self.dvh_status_label.setText("Load a patient folder to generate the axial view and filtered DVHs.")
        self.clear_dvh_curve_selection()
        self.update_dvh_cache_button()
        self.update_dose_range_controls()
        self.statusBar().clearMessage()
        self.clear_viewer_image_items()
        gc.collect()

    def clear_patient_queue_state(self) -> None:
        self.patient_preload_manager.cancel_all()
        self.patient_activation_prepare_manager.cancel_all()
        self.deferred_patient_view_refresh_token += 1
        self.patient_activation_token += 1
        self.patient_list_root_folder = None
        self.patient_list_folders = []
        self.current_patient_list_index = None
        self.pending_patient_preload_index = None
        self.pending_patient_preload_request_id = None
        self.pending_patient_activation_prepare_request_id = None
        self.pending_patient_activation_prepare_token = None
        self.pending_patient_activation_prepare_result = None
        self.pending_patient_activation_prepare_error = None
        self.preloaded_patient_index = None
        self.preloaded_patient_payload = None
        self.preloaded_patient_error = None
        blocker = QtCore.QSignalBlocker(self.patient_list_combo)
        self.patient_list_combo.clear()
        self.patient_list_combo.setEnabled(False)
        del blocker
        self.next_patient_button.setEnabled(False)

    def schedule_deferred_patient_view_refresh(
        self,
        *,
        timing_entries: Optional[List[Tuple[str, Optional[float]]]] = None,
        folder: Optional[str] = None,
    ) -> None:
        self.deferred_patient_view_refresh_token += 1
        refresh_token = self.deferred_patient_view_refresh_token

        def _run() -> None:
            if refresh_token != self.deferred_patient_view_refresh_token:
                return
            if folder is not None and self.current_patient_folder != folder:
                return
            if self.ct is None:
                return
            stage_start = perf_counter()
            self.refresh_orthogonal_views_from_controls()
            duration_s = perf_counter() - stage_start
            if timing_entries is not None:
                timing_entries.append(("Refresh orthogonal views (deferred)", duration_s))
                if self.latest_timing_folder == folder:
                    self.latest_timing_entries = list(timing_entries)
                    self.write_latest_timing_report()

        QtCore.QTimer.singleShot(0, _run)

    def get_patient_list_folder_entries(self, root_folder: str) -> List[str]:
        root_path = Path(root_folder)
        folders = [
            str(path)
            for path in sorted(root_path.iterdir(), key=lambda entry: entry.name.lower())
            if path.is_dir() and not path.name.startswith(".")
        ]
        return folders

    def populate_patient_list_combo(self) -> None:
        blocker = QtCore.QSignalBlocker(self.patient_list_combo)
        self.patient_list_combo.clear()
        for folder in self.patient_list_folders:
            folder_path = Path(folder)
            self.patient_list_combo.addItem(folder_path.name, folder)
        if self.current_patient_list_index is not None and 0 <= self.current_patient_list_index < self.patient_list_combo.count():
            self.patient_list_combo.setCurrentIndex(self.current_patient_list_index)
        self.patient_list_combo.setEnabled(bool(self.patient_list_folders))
        del blocker
        self.next_patient_button.setEnabled(self.preload_matches_next_patient())

    def can_advance_patient_queue(self) -> bool:
        return (
            self.current_patient_list_index is not None
            and (self.current_patient_list_index + 1) < len(self.patient_list_folders)
        )

    def get_next_patient_queue_index(self) -> Optional[int]:
        if not self.can_advance_patient_queue():
            return None
        assert self.current_patient_list_index is not None
        return self.current_patient_list_index + 1

    def preload_matches_next_patient(self) -> bool:
        next_index = self.get_next_patient_queue_index()
        return (
            next_index is not None
            and self.preloaded_patient_index == next_index
            and self.preloaded_patient_payload is not None
        )

    def update_patient_list_controls(self) -> None:
        self.patient_list_combo.setEnabled(bool(self.patient_list_folders))
        self.next_patient_button.setEnabled(self.preload_matches_next_patient())

    def start_background_preload_for_index(self, index: Optional[int]) -> None:
        self.patient_preload_manager.invalidate()
        self.pending_patient_preload_request_id = None
        self.pending_patient_preload_index = None
        self.preloaded_patient_index = None
        self.preloaded_patient_payload = None
        self.preloaded_patient_error = None
        if index is None or not (0 <= index < len(self.patient_list_folders)):
            self.update_patient_list_controls()
            return

        folder = self.patient_list_folders[index]
        self.pending_patient_preload_index = index
        self.pending_patient_preload_request_id = self.patient_preload_manager.start(
            folder,
            array_cache_signature=self.get_derived_array_cache_signature(),
        )
        self.statusBar().showMessage(f"Preloading next patient: {Path(folder).name}", 4000)
        self.update_patient_list_controls()

    def on_patient_preload_finished(
        self,
        request_id: int,
        folder: str,
        payload: object,
        duration_s: float,
    ) -> None:
        if not self.patient_preload_manager.is_current(request_id):
            return
        if not isinstance(payload, PatientPreloadPayload):
            return
        self.pending_patient_preload_request_id = None
        self.preloaded_patient_index = self.pending_patient_preload_index
        self.pending_patient_preload_index = None
        self.preloaded_patient_payload = payload
        self.preloaded_patient_error = None
        self.statusBar().showMessage(
            f"Next patient ready: {Path(folder).name} ({duration_s:.1f} s)",
            4000,
        )
        self.update_patient_list_controls()

    def on_patient_preload_failed(
        self,
        request_id: int,
        folder: str,
        error_message: str,
        duration_s: float,
    ) -> None:
        if not self.patient_preload_manager.is_current(request_id):
            return
        self.pending_patient_preload_request_id = None
        self.preloaded_patient_index = None
        self.pending_patient_preload_index = None
        self.preloaded_patient_payload = None
        self.preloaded_patient_error = error_message
        self.statusBar().showMessage(
            f"Background preload failed for {Path(folder).name}: {error_message}",
            8000,
        )
        self.update_patient_list_controls()

    def start_background_patient_activation_preparation(
        self,
        *,
        activation_token: int,
        folder: str,
        payload: PatientPreloadPayload,
    ) -> None:
        self.patient_activation_prepare_manager.cancel_all()
        self.pending_patient_activation_prepare_request_id = None
        self.pending_patient_activation_prepare_token = activation_token
        self.pending_patient_activation_prepare_result = None
        self.pending_patient_activation_prepare_error = None

        if payload.review_cache_data is None or self.rtstruct is None:
            self.pending_patient_activation_prepare_result = PatientActivationPreparationPayload(
                prepared_review_cache_state=None,
                cache_loaded=False,
                cache_load_duration=None,
                used_preloaded_review_cache=False,
            )
            return

        request_id = self.patient_activation_prepare_manager.start(
            folder,
            review_cache_data=payload.review_cache_data,
            expected_structure_names=[structure.name for structure in self.rtstruct.structures],
            available_constraint_sheet_names=self.available_constraint_sheet_names,
            no_constraints_sheet_label=NO_CONSTRAINTS_SHEET_LABEL,
            constraints_sheet_name=self.constraints_sheet_name,
            structure_filter_csv_path=self.structure_filter_csv_path,
            rtstruct_path=self.rtstruct_path,
            rtdose_paths=list(self.latest_timing_rtdose_paths),
            rtplan_paths=list(self.current_rtplan_paths),
            dvh_mode=self.get_dvh_mode(),
            dvh_method_signature=get_dvh_method_signature(),
            target_method_signature=self.get_target_method_signature(),
            has_ct=self.ct is not None,
            has_dose=self.dose is not None,
        )
        self.pending_patient_activation_prepare_request_id = request_id

    def on_patient_activation_prepare_finished(
        self,
        request_id: int,
        folder: str,
        payload: object,
        duration_s: float,
    ) -> None:
        if request_id != self.pending_patient_activation_prepare_request_id:
            return
        if self.current_patient_folder != folder:
            return
        if not isinstance(payload, PatientActivationPreparationPayload):
            return
        self.pending_patient_activation_prepare_request_id = None
        self.pending_patient_activation_prepare_result = payload
        self.pending_patient_activation_prepare_error = None
        self.statusBar().showMessage(
            f"Prepared final review state for {Path(folder).name} ({duration_s:.1f} s)",
            2000,
        )

    def on_patient_activation_prepare_failed(
        self,
        request_id: int,
        folder: str,
        error_message: str,
        _duration_s: float,
    ) -> None:
        if request_id != self.pending_patient_activation_prepare_request_id:
            return
        if self.current_patient_folder != folder:
            return
        self.pending_patient_activation_prepare_request_id = None
        self.pending_patient_activation_prepare_result = None
        self.pending_patient_activation_prepare_error = error_message

    def on_load_patient_list_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Patient List Folder")
        if not folder:
            return
        self.load_patient_list_folder(folder)

    def load_patient_list_folder(self, folder: str) -> None:
        patient_folders = self.get_patient_list_folder_entries(folder)
        if not patient_folders:
            QtWidgets.QMessageBox.information(
                self,
                "No patient folders",
                "The selected folder did not contain any patient subfolders.",
            )
            return
        self.clear_patient_queue_state()
        self.patient_list_root_folder = folder
        self.patient_list_folders = patient_folders
        self.current_patient_list_index = 0
        self.populate_patient_list_combo()
        self.load_patient_folder_path(patient_folders[0], patient_list_index=0)

    def on_patient_list_combo_activated(self, index: int) -> None:
        if not (0 <= index < len(self.patient_list_folders)):
            return
        payload = self.preloaded_patient_payload if self.preloaded_patient_index == index else None
        self.load_patient_folder_path(
            self.patient_list_folders[index],
            patient_list_index=index,
            preloaded_payload=payload,
        )

    def on_next_patient_clicked(self) -> None:
        next_index = self.get_next_patient_queue_index()
        if next_index is None:
            return
        payload = self.preloaded_patient_payload if self.preloaded_patient_index == next_index else None
        if payload is None and self.pending_patient_preload_index == next_index:
            self.statusBar().showMessage(
                f"Next patient is still loading: {Path(self.patient_list_folders[next_index]).name}",
                4000,
            )
            return
        if payload is None:
            self.statusBar().showMessage(
                f"Next patient is not ready yet: {Path(self.patient_list_folders[next_index]).name}",
                4000,
            )
            return
        self.load_patient_folder_path(
            self.patient_list_folders[next_index],
            patient_list_index=next_index,
            preloaded_payload=payload,
        )

    def _load_saved_review_cache_if_available(
        self,
        *,
        derived_array_cache_path: Optional[Path],
        preloaded_review_cache_data: Optional[ReviewCacheFileData] = None,
    ) -> Tuple[ReviewCacheAvailability, bool, Optional[float], bool]:
        cache_info = get_review_cache_availability(
            dvh_can_start=self.ct is not None and self.dose is not None and self.rtstruct is not None,
            cache_path=self.get_dvh_cache_path(),
            derived_array_cache_path=derived_array_cache_path,
        )
        cache_loaded = False
        cache_load_duration: Optional[float] = None
        used_preloaded_review_cache = False
        if cache_info.dvh_can_start:
            if preloaded_review_cache_data is not None:
                self.show_progress_status("Using preloaded saved JSON cache", pump_events=True)
            elif cache_info.cache_found:
                self.show_progress_status("Found saved JSON cache", pump_events=True)
            elif cache_info.derived_sidecar_only:
                self.show_progress_status("Found saved derived cache only; no saved JSON review data", pump_events=True)
            stage_start = perf_counter()
            self.restore_saved_results_without_calculation = True
            if preloaded_review_cache_data is not None:
                cache_loaded = self.load_saved_review_cache_data(preloaded_review_cache_data, refresh_ui=False)
                used_preloaded_review_cache = cache_loaded
            else:
                cache_loaded = self.try_load_saved_dvh_cache(refresh_ui=False)
            if not cache_loaded:
                self.restore_saved_results_without_calculation = False
            cache_load_duration = perf_counter() - stage_start if cache_loaded else None
        return cache_info, cache_loaded, cache_load_duration, used_preloaded_review_cache

    def _finalize_patient_load_interactive_state(
        self,
        *,
        cache_info: ReviewCacheAvailability,
        cache_loaded: bool,
        cache_load_duration: Optional[float],
        used_preloaded_review_cache: bool,
        fast_activate: bool,
        patient_folder: Optional[str],
        timing_entries: List[Tuple[str, Optional[float]]],
        overall_start: float,
    ) -> None:
        self.defer_sidebar_summary_metrics = False
        if self.rtstruct is not None and self.ct is not None:
            if cache_loaded:
                self.populate_structures_list()
                stage_start = perf_counter()
                self.update_display()
                self.apply_image_based_view_ranges()
                timing_entries.append(("Refresh axial view (final)", perf_counter() - stage_start))
                self.render_dvh_plot()
                if self.dvh_curves:
                    self.dvh_status_label.setText("Loaded saved DVH/constraints cache.")
                else:
                    self.dvh_status_label.setText("Saved DVH cache contained no curves.")
                self.update_dvh_cache_button()
                self.schedule_deferred_patient_view_refresh(
                    timing_entries=timing_entries,
                    folder=patient_folder,
                )
            else:
                if cache_info.cache_found:
                    self.show_progress_status("Saved JSON cache found but not usable; recalculating", pump_events=True)
                elif cache_info.derived_sidecar_only:
                    self.show_progress_status("Using derived cache only; recalculating review state", pump_events=True)
                self.show_progress_status("Computing metrics", pump_events=True)
                self.populate_structures_list()
                stage_start = perf_counter()
                self.update_display()
                self.apply_image_based_view_ranges()
                self.render_dvh_plot()
                timing_entries.append(("Refresh axial view (final)", perf_counter() - stage_start))
                self.schedule_deferred_patient_view_refresh(
                    timing_entries=timing_entries,
                    folder=patient_folder,
                )
        elif self.ct is not None:
            stage_start = perf_counter()
            self.update_display()
            self.apply_image_based_view_ranges()
            timing_entries.append(("Refresh axial view (final)", perf_counter() - stage_start))
            self.schedule_deferred_patient_view_refresh(
                timing_entries=timing_entries,
                folder=patient_folder,
            )

        if cache_loaded:
            if used_preloaded_review_cache:
                self.show_progress_status("Loaded preloaded saved JSON cache")
            else:
                self.show_progress_status("Loaded saved JSON cache")
            self.restore_saved_results_without_calculation = False
        timing_entries.append(("Load saved DVH cache", cache_load_duration))
        if not cache_loaded:
            self.restore_saved_results_without_calculation = False
            timing_entries.append(("Compute DVH (background)" if cache_info.dvh_can_start else "Compute DVH", None))
        timing_entries.append(("Total patient load to interactive review", perf_counter() - overall_start))

    def _apply_patient_preload_payload(
        self,
        payload: PatientPreloadPayload,
        *,
        timing_entries: List[Tuple[str, Optional[float]]],
    ) -> Tuple[Optional[str], Optional[str], List[str], Optional[Path]]:
        self.ct = payload.ct
        self.plan_phases = list(payload.plan_phases)
        self.current_rtplan_paths = list(payload.rtplan_paths)
        self.set_patient_plan_lines(payload.patient_plan_lines, pump_events=True)
        self.image_view_bounds = payload.image_view_bounds
        self.rtstruct_path = payload.rtstruct_path
        self.rtstruct = payload.rtstruct
        self.dose = payload.dose
        self.sampled_dose_volume_ct = payload.sampled_dose_volume_ct
        self.latest_timing_rtdose_paths = list(payload.rtdose_paths)

        self.displayed_dose_plane = None
        self.structure_mask_cache = None
        self.structure_mask_cache_names = []
        self.max_dose_index_zyx = None
        self.current_row = self.ct.rows // 2
        self.current_col = self.ct.cols // 2
        self.reset_autoscroll_speed()
        self.slice_slider.setRange(0, self.ct.volume_hu.shape[0] - 1)
        self.slice_slider.setValue(self.ct.volume_hu.shape[0] // 2)
        self.update_autoscroll_speed_label()

        self.refresh_constraint_sheet_combo(preferred_sheet_name=None)
        constraints_path = self.structure_filter_csv_path
        if constraints_path is not None and self.constraints_sheet_name is not None:
            stage_start = perf_counter()
            (
                _,
                self.csv_structure_goals_by_name,
                self.structure_csv_order,
            ) = load_structure_constraints_sheet(
                constraints_path,
                self.constraints_sheet_name,
                self.plan_phases,
            )
            self.rebuild_structure_goals_by_name()
            timing_entries.append(("Load constraints workbook", perf_counter() - stage_start))
        else:
            self.csv_structure_goals_by_name = {}
            self.structure_csv_order = []
            self.rebuild_structure_goals_by_name()
            timing_entries.append(("Load constraints workbook", None))

        if self.rtstruct is not None:
            self.sort_rtstruct_structures_for_display()
            visible_range = self.get_visible_structure_slice_range()
            if visible_range is not None:
                start_idx, end_idx = visible_range
                self.slice_slider.setValue((start_idx + end_idx) // 2)
        else:
            self.populate_structures_list()
            self.populate_isodose_controls()

        if payload.derived_array_cache_data is not None:
            self.ptv_union_volume_mask_cache = payload.derived_array_cache_data.ptv_union_volume_mask
            self.structure_volume_mask_cache = dict(payload.derived_array_cache_data.structure_volume_masks)
            self.structure_geometry_volume_cache = dict(payload.derived_array_cache_data.structure_geometry_volumes_cc)

        if self.dose is not None:
            self.target_curve_cache = {}
            self.target_metrics_cache = {}
            self.stereotactic_metrics_cache = {}
            self.stereotactic_volume_context_cache = {}
            self.max_tissue_dose_gy_cache = None
            self.max_tissue_index_zyx = None
            self.cached_target_table_rows = None
            self.apply_default_dose_range()
        if self.rtstruct is not None:
            self.populate_isodose_controls()

        timing_entries.extend(payload.timing_entries)
        derived_array_cache_path = self.get_derived_array_cache_path()
        return constraints_path, self.rtstruct_path, list(payload.rtdose_paths), derived_array_cache_path

    def apply_precomputed_initial_view_state(
        self,
        precomputed_view_state: PrecomputedPatientViewState,
    ) -> None:
        blocker = QtCore.QSignalBlocker(self.slice_slider)
        self.slice_slider.setValue(precomputed_view_state.slice_index)
        del blocker
        self.current_row = precomputed_view_state.row_idx
        self.current_col = precomputed_view_state.col_idx

        ww = precomputed_view_state.window_width
        wl = precomputed_view_state.window_level
        lo = wl - ww / 2.0
        hi = wl + ww / 2.0
        blocker = QtCore.QSignalBlocker(self.window_level_slider)
        self.window_level_slider.setWindowLevel(int(round(ww)), int(round(wl)))
        del blocker
        self.window_label.setText(f"WL/WW: {int(round(wl))} / {int(round(ww))}")

        if self.dose is not None:
            self.dose_overlay_enabled = precomputed_view_state.dose_alpha > 0.0
            blocker = QtCore.QSignalBlocker(self.dose_toggle_button)
            self.dose_toggle_button.setChecked(self.dose_overlay_enabled)
            del blocker
            self.dose_toggle_button.setText("On" if self.dose_overlay_enabled else "Off")
            blocker = QtCore.QSignalBlocker(self.dose_opacity_slider)
            self.dose_opacity_slider.setValue(int(round(precomputed_view_state.dose_alpha * 100.0)))
            del blocker
            lower_value = self.dose_gy_to_slider_value(precomputed_view_state.dose_min_gy)
            upper_value = self.dose_gy_to_slider_value(precomputed_view_state.dose_max_gy)
            blocker = QtCore.QSignalBlocker(self.dose_range_slider)
            self.dose_range_slider.setValues(lower_value, upper_value)
            del blocker

        axial_state = precomputed_view_state.axial_render_state
        self.ct_item.setImage(axial_state.ct_plane, levels=(lo, hi), autoLevels=False)
        self.displayed_dose_plane = axial_state.dose_plane
        self.dose_item.setImage(axial_state.dose_rgba, autoLevels=False)
        if axial_state.dose_plane is not None:
            self.add_isodose_items(self.axial_view, self.axial_isodose_items, axial_state.dose_plane)
        else:
            self.clear_overlay_items(self.axial_view, self.axial_isodose_items)
        self.clear_overlay_items(self.axial_view, self.axial_contour_items)
        apply_polyline_specs(self.axial_view, self.axial_contour_items, axial_state.contour_specs)
        self.slice_label.setText(axial_state.slice_label_text)
        self.z_label.setText(axial_state.z_label_text)

        orth_state = precomputed_view_state.orthogonal_render_state
        self.sagittal_ct_item.setImage(orth_state.sagittal_plane, levels=(lo, hi), autoLevels=False)
        self.coronal_ct_item.setImage(orth_state.coronal_plane, levels=(lo, hi), autoLevels=False)
        self.sagittal_dose_item.setImage(orth_state.sagittal_dose_rgba, autoLevels=False)
        self.coronal_dose_item.setImage(orth_state.coronal_dose_rgba, autoLevels=False)
        if orth_state.sagittal_dose_plane is not None and orth_state.coronal_dose_plane is not None:
            self.add_isodose_items(self.sagittal_view, self.sagittal_isodose_items, orth_state.sagittal_dose_plane)
            self.add_isodose_items(self.coronal_view, self.coronal_isodose_items, orth_state.coronal_dose_plane)
        else:
            self.clear_overlay_items(self.sagittal_view, self.sagittal_isodose_items)
            self.clear_overlay_items(self.coronal_view, self.coronal_isodose_items)
        self.clear_overlay_items(self.sagittal_view, self.sagittal_contour_items)
        self.clear_overlay_items(self.coronal_view, self.coronal_contour_items)
        apply_polyline_specs(self.sagittal_view, self.sagittal_contour_items, orth_state.sagittal_contours)
        apply_polyline_specs(self.coronal_view, self.coronal_contour_items, orth_state.coronal_contours)

        self.update_dose_range_controls()
        self.update_max_dose_markers()
        self.update_axial_overlay_positions()
        self.apply_image_based_view_ranges()

    def apply_precomputed_axial_render_state(
        self,
        *,
        slice_index: int,
        axial_state: object,
    ) -> None:
        if self.ct is None:
            return
        ww, wl = self.get_window_level()
        lo = wl - ww / 2.0
        hi = wl + ww / 2.0
        blocker = QtCore.QSignalBlocker(self.slice_slider)
        self.slice_slider.setValue(slice_index)
        del blocker
        self.ct_item.setImage(axial_state.ct_plane, levels=(lo, hi), autoLevels=False)
        self.displayed_dose_plane = axial_state.dose_plane
        self.dose_item.setImage(axial_state.dose_rgba, autoLevels=False)
        if axial_state.dose_plane is not None:
            self.add_isodose_items(self.axial_view, self.axial_isodose_items, axial_state.dose_plane)
        else:
            self.clear_overlay_items(self.axial_view, self.axial_isodose_items)
        self.clear_overlay_items(self.axial_view, self.axial_contour_items)
        apply_polyline_specs(self.axial_view, self.axial_contour_items, axial_state.contour_specs)
        self.slice_label.setText(axial_state.slice_label_text)
        self.z_label.setText(axial_state.z_label_text)
        self.window_label.setText(axial_state.window_label_text)
        self.update_axial_overlay_positions()

    def build_patient_transition_overlay_geometry(self) -> Optional[Tuple[QtCore.QRect, QtCore.QRect]]:
        if not hasattr(self, "axial_tab_widget") or not hasattr(self, "axial_graphics_widget"):
            return None
        overlay_widget = self.axial_tab_widget
        axial_widget = self.axial_graphics_widget
        overlay_size = overlay_widget.size()
        axial_size = axial_widget.size()
        if overlay_size.width() <= 0 or overlay_size.height() <= 0:
            return None
        if axial_size.width() <= 0 or axial_size.height() <= 0:
            return None
        overlay_top_left = overlay_widget.mapToGlobal(QtCore.QPoint(0, 0))
        overlay_rect = QtCore.QRect(overlay_top_left, overlay_size)
        axial_top_left = axial_widget.mapTo(overlay_widget, QtCore.QPoint(0, 0))
        axial_rect = QtCore.QRect(axial_top_left, axial_size)
        return overlay_rect, axial_rect

    def write_review_movie_asset(
        self,
        path: Path,
        frames_rgba: List[np.ndarray],
        interval_ms: int,
    ) -> None:
        if not frames_rgba:
            raise ValueError("No cine frames are available to save.")
        frames = np.stack(
            [np.ascontiguousarray(frame, dtype=np.uint8) for frame in frames_rgba],
            axis=0,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path_str = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=path.suffix, dir=path.parent)
        temp_path = Path(temp_path_str)
        os.close(fd)
        try:
            try:
                temp_path.unlink()
            except OSError:
                pass
            np.savez(
                temp_path,
                frames=frames,
                interval_ms=np.asarray([max(60, int(interval_ms))], dtype=np.int32),
            )
            temp_path.replace(path)
        except Exception:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def get_review_movie_path(self, base_path: Optional[Path] = None) -> Optional[Path]:
        cache_path = base_path if base_path is not None else self.get_dvh_cache_path()
        if cache_path is None:
            return None
        return cache_path.with_name(f"{cache_path.stem}_movie.npz")

    def save_review_movie(self, path: Path) -> Optional[str]:
        if self.ct is None:
            return "failed to save the review movie."
        cine_indices, cine_interval_ms = build_axial_cine_plan(self.ct, self.rtstruct)
        if not cine_indices:
            return "failed to save the review movie."
        if not hasattr(self, "axial_graphics_widget"):
            return "failed to save the review movie."

        current_tab = self.tabs.currentWidget()
        original_slice_index = int(np.clip(self.slice_slider.value(), 0, self.ct.volume_hu.shape[0] - 1))
        original_view_range = self.axial_view.viewRange()
        original_readout_visible = self.axial_readout_label.isVisible()
        original_autoscroll_visible = self.axial_autoscroll_overlay.isVisible()
        original_crosshair_visible = self.crosshair_text.isVisible()
        original_marker_visible = self.axial_max_marker.isVisible()
        was_autoscrolling = self.autoscroll_button.isChecked()
        ww, wl = self.get_window_level()
        lo = wl - ww / 2.0
        hi = wl + ww / 2.0
        min_dose, max_dose = self.get_dose_display_range()
        dose_alpha = self.current_dose_alpha()
        captured_frames: List[np.ndarray] = []
        try:
            if was_autoscrolling:
                self.cancel_autoscroll()
            self.tabs.setCurrentWidget(self.axial_tab_widget)
            self.axial_readout_label.hide()
            self.axial_autoscroll_overlay.hide()
            self.crosshair_text.hide()
            self.axial_max_marker.hide()
            self.pump_viewer_ui()

            for frame_slice_index in cine_indices:
                frame_state = build_axial_render_state(
                    self.ct,
                    self.dose,
                    self.rtstruct,
                    self.sampled_dose_volume_ct,
                    frame_slice_index,
                    lo,
                    hi,
                    dose_alpha,
                    min_dose,
                    max_dose,
                    self.structure_is_visible,
                )
                self.apply_precomputed_axial_render_state(
                    slice_index=frame_slice_index,
                    axial_state=frame_state,
                )
                self.axial_view.setRange(
                    xRange=(float(original_view_range[0][0]), float(original_view_range[0][1])),
                    yRange=(float(original_view_range[1][0]), float(original_view_range[1][1])),
                    padding=0.0,
                )
                self.pump_viewer_ui()
                pixmap = self.axial_graphics_widget.grab()
                if pixmap.isNull():
                    raise RuntimeError("failed to capture an axial movie frame")
                image = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
                width = image.width()
                height = image.height()
                buffer = image.bits()
                frame_rgba = np.frombuffer(buffer, dtype=np.uint8, count=height * width * 4).reshape((height, width, 4)).copy()
                captured_frames.append(frame_rgba)

            self.write_review_movie_asset(path, captured_frames, cine_interval_ms)
        except Exception as exc:
            logger.warning("Saved JSON review cache but failed to save review movie: %s", exc)
            return f"failed to save the review movie: {exc}"
        finally:
            restore_state = build_axial_render_state(
                self.ct,
                self.dose,
                self.rtstruct,
                self.sampled_dose_volume_ct,
                original_slice_index,
                lo,
                hi,
                dose_alpha,
                min_dose,
                max_dose,
                self.structure_is_visible,
            )
            self.apply_precomputed_axial_render_state(
                slice_index=original_slice_index,
                axial_state=restore_state,
            )
            self.axial_view.setRange(
                xRange=(float(original_view_range[0][0]), float(original_view_range[0][1])),
                yRange=(float(original_view_range[1][0]), float(original_view_range[1][1])),
                padding=0.0,
            )
            if original_readout_visible:
                self.axial_readout_label.show()
            if original_autoscroll_visible:
                self.axial_autoscroll_overlay.show()
            if original_crosshair_visible:
                self.crosshair_text.show()
            if original_marker_visible:
                self.axial_max_marker.show()
            if current_tab is not None:
                self.tabs.setCurrentWidget(current_tab)
            if was_autoscrolling:
                blocker = QtCore.QSignalBlocker(self.autoscroll_button)
                self.autoscroll_button.setChecked(True)
                del blocker
                self.set_autoscroll_ui_locked(True)
                self.autoscroll_timer.start()
            self.pump_viewer_ui()
        return None

    def start_patient_transition_overlay_process(
        self,
        folder: str,
    ) -> bool:
        self.stop_patient_transition_overlay_process()

        geometry = self.build_patient_transition_overlay_geometry()
        cache_path = compute_dvh_cache_path(folder)
        screenshot_path = self.get_review_screenshot_path(cache_path)
        saved_movie_path = self.get_review_movie_path(cache_path)
        has_screenshot = screenshot_path is not None and screenshot_path.exists()
        has_saved_movie = saved_movie_path is not None and saved_movie_path.exists()
        movie_path: Optional[Path] = saved_movie_path if has_saved_movie else None

        if geometry is None or (not has_screenshot and movie_path is None):
            return False

        overlay_rect, axial_rect = geometry
        overlay_script = Path(__file__).with_name("peer_transition_overlay.py")
        command = [
            sys.executable,
            str(overlay_script),
            "--window-x",
            str(overlay_rect.x()),
            "--window-y",
            str(overlay_rect.y()),
            "--window-width",
            str(overlay_rect.width()),
            "--window-height",
            str(overlay_rect.height()),
            "--axial-x",
            str(axial_rect.x()),
            "--axial-y",
            str(axial_rect.y()),
            "--axial-width",
            str(axial_rect.width()),
            "--axial-height",
            str(axial_rect.height()),
            "--parent-pid",
            str(os.getpid()),
        ]
        if has_screenshot and screenshot_path is not None:
            command.extend(["--screenshot", str(screenshot_path)])
        if movie_path is not None:
            command.extend(["--movie", str(movie_path)])
        try:
            self.patient_transition_overlay_process = subprocess.Popen(
                command,
                cwd=str(overlay_script.parent),
                stdin=subprocess.DEVNULL,
            )
            self.patient_transition_overlay_temp_dir = None
        except Exception:
            self.patient_transition_overlay_process = None
            self.patient_transition_overlay_temp_dir = None
            return False
        return True

    def stop_patient_transition_overlay_process(self) -> None:
        process = self.patient_transition_overlay_process
        self.patient_transition_overlay_process = None
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=0.75)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=0.5)
            except Exception:
                pass
        temp_dir = self.patient_transition_overlay_temp_dir
        self.patient_transition_overlay_temp_dir = None
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def pump_viewer_ui(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def get_patient_activation_midstage_delay_ms(self) -> int:
        return max(900, int(self.autoscroll_timer.interval()) * 6)

    def get_patient_activation_step_delay_ms(self) -> int:
        return max(90, int(self.autoscroll_timer.interval()))

    def load_patient_folder_path(
        self,
        folder: str,
        *,
        patient_list_index: Optional[int] = None,
        preloaded_payload: Optional[PatientPreloadPayload] = None,
    ) -> bool:
        timing_entries: List[Tuple[str, Optional[float]]] = []
        overall_start = perf_counter()
        fast_activate = preloaded_payload is not None and patient_list_index is not None
        constraints_path: Optional[str] = None
        rtstruct_path: Optional[str] = None
        rtdose_paths: List[str] = []
        derived_array_cache_path: Optional[Path] = None

        self.patient_preload_manager.invalidate()
        self.pending_patient_preload_request_id = None
        self.pending_patient_preload_index = None
        self.preloaded_patient_index = None
        self.preloaded_patient_payload = None
        self.preloaded_patient_error = None

        self.set_heavy_view_updates_enabled(fast_activate)
        try:
            self.show_progress_status(
                "Activating preloaded patient..." if preloaded_payload is not None else "Scanning patient folder...",
                pump_events=True,
            )
            self.clear_patient_session_state(quick_swap=fast_activate)
            self.current_patient_folder = folder
            self.defer_sidebar_summary_metrics = True
            if preloaded_payload is not None and preloaded_payload.patient_plan_lines:
                self.set_patient_plan_lines(preloaded_payload.patient_plan_lines, pump_events=True)
            elif preloaded_payload is None:
                self.try_stage_patient_plan_lines_from_cache(folder, pump_events=True)
            if preloaded_payload is None:
                self.tabs.setCurrentWidget(self.axial_tab_widget)
                self.start_patient_transition_overlay_process(folder)
                self.pump_viewer_ui()

            if preloaded_payload is None:
                payload = prepare_patient_preload_payload(
                    folder,
                    array_cache_signature=self.get_derived_array_cache_signature(),
                    progress_callback=lambda message: self.show_progress_status(message, pump_events=True),
                    patient_plan_callback=lambda lines: self.set_patient_plan_lines(lines, pump_events=True),
                    include_precomputed_view_state=False,
                )
            else:
                payload = preloaded_payload

            constraints_path, rtstruct_path, rtdose_paths, derived_array_cache_path = self._apply_patient_preload_payload(
                payload,
                timing_entries=timing_entries,
            )

            stage_start = perf_counter()
            if fast_activate and payload.precomputed_view_state is not None:
                self.tabs.setCurrentWidget(self.axial_tab_widget)
                self.apply_precomputed_initial_view_state(payload.precomputed_view_state)
                timing_entries.append(("Apply precomputed initial views", perf_counter() - stage_start))
                self.pump_viewer_ui()
            else:
                self.update_display()
                self.apply_image_based_view_ranges()
                timing_entries.append(("Refresh axial view", perf_counter() - stage_start))

            self.latest_timing_entries = list(timing_entries)
            self.latest_timing_folder = folder
            self.latest_timing_csv_path = constraints_path
            self.latest_timing_rtstruct_path = rtstruct_path
            self.latest_timing_rtdose_paths = list(rtdose_paths)
            if fast_activate:
                self.show_progress_status("Finalizing preloaded patient...", pump_events=False)
                self.set_patient_activation_ui_locked(True)
                if payload.precomputed_view_state is not None:
                    self.start_patient_transition_overlay_process(folder)
                self.pump_viewer_ui()
                activation_token = self.patient_activation_token + 1
                self.patient_activation_token = activation_token
                self.start_background_patient_activation_preparation(
                    activation_token=activation_token,
                    folder=folder,
                    payload=payload,
                )

                def _continue_activation() -> None:
                    self._continue_patient_activation_when_ready(
                        activation_token=activation_token,
                        folder=folder,
                        patient_list_index=patient_list_index,
                        constraints_path=constraints_path,
                        rtstruct_path=rtstruct_path,
                        rtdose_paths=rtdose_paths,
                        timing_entries=timing_entries,
                        overall_start=overall_start,
                    )

                QtCore.QTimer.singleShot(self.get_patient_activation_midstage_delay_ms(), _continue_activation)
                return True

            return self._complete_patient_folder_activation(
                activation_token=self.patient_activation_token,
                folder=folder,
                patient_list_index=patient_list_index,
                payload=payload,
                constraints_path=constraints_path,
                rtstruct_path=rtstruct_path,
                rtdose_paths=rtdose_paths,
                derived_array_cache_path=derived_array_cache_path,
                timing_entries=timing_entries,
                overall_start=overall_start,
                fast_activate=fast_activate,
            )
        except Exception as e:
            self.clear_patient_session_state()
            timing_entries.append(("Total patient load to interactive review (failed)", perf_counter() - overall_start))
            self.write_load_timing_report(
                folder,
                timing_entries,
                csv_path=constraints_path,
                rtstruct_path=rtstruct_path,
                rtdose_paths=rtdose_paths,
                error_message=str(e),
            )
            QtWidgets.QMessageBox.critical(self, "Patient load failed", str(e))
            return False
        finally:
            if not fast_activate:
                self.set_patient_activation_ui_locked(False)
                self.set_heavy_view_updates_enabled(True)

    def _is_current_patient_activation(self, activation_token: int, folder: str) -> bool:
        return activation_token == self.patient_activation_token and self.current_patient_folder == folder

    def _handle_patient_activation_failure(
        self,
        *,
        activation_token: int,
        folder: str,
        constraints_path: Optional[str],
        rtstruct_path: Optional[str],
        rtdose_paths: List[str],
        timing_entries: List[Tuple[str, Optional[float]]],
        overall_start: float,
        error: Exception,
    ) -> bool:
        if self._is_current_patient_activation(activation_token, folder):
            self.clear_patient_session_state()
        failed_timing_entries = list(timing_entries)
        failed_timing_entries.append(("Total patient load to interactive review (failed)", perf_counter() - overall_start))
        self.write_load_timing_report(
            folder,
            failed_timing_entries,
            csv_path=constraints_path,
            rtstruct_path=rtstruct_path,
            rtdose_paths=rtdose_paths,
            error_message=str(error),
        )
        QtWidgets.QMessageBox.critical(self, "Patient load failed", str(error))
        self.stop_patient_transition_overlay_process()
        self.set_patient_activation_ui_locked(False)
        self.set_heavy_view_updates_enabled(True)
        self.restore_saved_results_without_calculation = False
        return False

    def _finish_patient_activation_success(
        self,
        *,
        activation_token: int,
        folder: str,
        patient_list_index: Optional[int],
        constraints_path: Optional[str],
        rtstruct_path: Optional[str],
        rtdose_paths: List[str],
        timing_entries: List[Tuple[str, Optional[float]]],
        cache_info: ReviewCacheAvailability,
        cache_loaded: bool,
    ) -> bool:
        if not self._is_current_patient_activation(activation_token, folder):
            return False

        self.latest_timing_entries = list(timing_entries)
        self.latest_timing_folder = folder
        self.latest_timing_csv_path = constraints_path
        self.latest_timing_rtstruct_path = rtstruct_path
        self.latest_timing_rtdose_paths = list(rtdose_paths)

        self.write_latest_timing_report()

        dvh_started = False
        if not cache_loaded:
            dvh_started = self.refresh_dvh()
            if not dvh_started and cache_info.dvh_can_start:
                self.latest_timing_entries = [
                    ("Compute DVH", duration_s) if label == "Compute DVH (background)" else (label, duration_s)
                    for label, duration_s in self.latest_timing_entries
                ]
                self.write_latest_timing_report()
        if cache_loaded or not dvh_started:
            self.clear_progress_status()

        if self.constraint_workbook_error:
            self.statusBar().showMessage(
                f"Could not read constraints workbook: {self.constraint_workbook_error}",
                8000,
            )
        else:
            self.statusBar().clearMessage()

        if patient_list_index is not None:
            self.current_patient_list_index = patient_list_index
            self.populate_patient_list_combo()
            self.start_background_preload_for_index(self.get_next_patient_queue_index())
        else:
            self.update_patient_list_controls()

        self.stop_patient_transition_overlay_process()
        self.set_patient_activation_ui_locked(False)
        self.set_heavy_view_updates_enabled(True)
        self.restore_saved_results_without_calculation = False
        return True

    def _continue_patient_activation_when_ready(
        self,
        *,
        activation_token: int,
        folder: str,
        patient_list_index: Optional[int],
        constraints_path: Optional[str],
        rtstruct_path: Optional[str],
        rtdose_paths: List[str],
        timing_entries: List[Tuple[str, Optional[float]]],
        overall_start: float,
    ) -> bool:
        if not self._is_current_patient_activation(activation_token, folder):
            return False
        if self.pending_patient_activation_prepare_token != activation_token:
            return False
        if self.pending_patient_activation_prepare_error:
            return self._handle_patient_activation_failure(
                activation_token=activation_token,
                folder=folder,
                constraints_path=constraints_path,
                rtstruct_path=rtstruct_path,
                rtdose_paths=rtdose_paths,
                timing_entries=timing_entries,
                overall_start=overall_start,
                error=RuntimeError(self.pending_patient_activation_prepare_error),
            )
        if self.pending_patient_activation_prepare_result is None:
            QtCore.QTimer.singleShot(
                self.get_patient_activation_step_delay_ms(),
                lambda: self._continue_patient_activation_when_ready(
                    activation_token=activation_token,
                    folder=folder,
                    patient_list_index=patient_list_index,
                    constraints_path=constraints_path,
                    rtstruct_path=rtstruct_path,
                    rtdose_paths=rtdose_paths,
                    timing_entries=timing_entries,
                    overall_start=overall_start,
                ),
            )
            return True

        prepared_activation = self.pending_patient_activation_prepare_result
        self.pending_patient_activation_prepare_request_id = None
        self.pending_patient_activation_prepare_token = None
        self.pending_patient_activation_prepare_result = None
        self.pending_patient_activation_prepare_error = None
        return self._start_chunked_patient_folder_activation(
            activation_token=activation_token,
            folder=folder,
            patient_list_index=patient_list_index,
            constraints_path=constraints_path,
            rtstruct_path=rtstruct_path,
            rtdose_paths=rtdose_paths,
            timing_entries=timing_entries,
            overall_start=overall_start,
            activation_preparation=prepared_activation,
        )

    def _run_patient_activation_step_sequence(
        self,
        *,
        activation_token: int,
        folder: str,
        steps: List[Callable[[], None]],
        constraints_path: Optional[str],
        rtstruct_path: Optional[str],
        rtdose_paths: List[str],
        timing_entries: List[Tuple[str, Optional[float]]],
        overall_start: float,
    ) -> bool:
        if not self._is_current_patient_activation(activation_token, folder):
            return False
        if not steps:
            return True
        step = steps[0]
        remaining_steps = steps[1:]
        try:
            step()
        except Exception as error:
            return self._handle_patient_activation_failure(
                activation_token=activation_token,
                folder=folder,
                constraints_path=constraints_path,
                rtstruct_path=rtstruct_path,
                rtdose_paths=rtdose_paths,
                timing_entries=timing_entries,
                overall_start=overall_start,
                error=error,
            )
        if remaining_steps:
            self.pump_viewer_ui()
            QtCore.QTimer.singleShot(
                self.get_patient_activation_step_delay_ms(),
                lambda: self._run_patient_activation_step_sequence(
                    activation_token=activation_token,
                    folder=folder,
                    steps=remaining_steps,
                    constraints_path=constraints_path,
                    rtstruct_path=rtstruct_path,
                    rtdose_paths=rtdose_paths,
                    timing_entries=timing_entries,
                    overall_start=overall_start,
                ),
            )
        return True

    def _start_chunked_patient_folder_activation(
        self,
        *,
        activation_token: int,
        folder: str,
        patient_list_index: Optional[int],
        constraints_path: Optional[str],
        rtstruct_path: Optional[str],
        rtdose_paths: List[str],
        timing_entries: List[Tuple[str, Optional[float]]],
        overall_start: float,
        activation_preparation: PatientActivationPreparationPayload,
    ) -> bool:
        if not self._is_current_patient_activation(activation_token, folder):
            return False

        self.restore_saved_results_without_calculation = bool(activation_preparation.cache_loaded)
        cache_info = get_review_cache_availability(
            dvh_can_start=self.ct is not None and self.dose is not None and self.rtstruct is not None,
            cache_path=self.get_dvh_cache_path(),
            derived_array_cache_path=self.get_derived_array_cache_path(),
        )
        state: Dict[str, object] = {
            "cache_info": cache_info,
            "cache_loaded": activation_preparation.cache_loaded,
            "cache_load_duration": activation_preparation.cache_load_duration,
            "used_preloaded_review_cache": activation_preparation.used_preloaded_review_cache,
            "prepared_review_cache_state": activation_preparation.prepared_review_cache_state,
        }

        def step_prepare_status() -> None:
            self.defer_sidebar_summary_metrics = False

            if self.rtstruct is None or self.ct is None:
                return

            if not bool(state["cache_loaded"]):
                cache_info = state["cache_info"]
                if cache_info.cache_found:
                    self.show_progress_status("Saved JSON cache found but not usable; recalculating", pump_events=False)
                elif cache_info.derived_sidecar_only:
                    self.show_progress_status("Using derived cache only; recalculating review state", pump_events=False)
                self.show_progress_status("Computing metrics", pump_events=False)

        def step_populate_structures() -> None:
            if self.rtstruct is not None and self.ct is not None:
                self.populate_structures_list()

        def step_refresh_axial_final() -> None:
            if self.ct is None:
                return
            stage_start = perf_counter()
            self.update_display()
            self.apply_image_based_view_ranges()
            timing_entries.append(("Refresh axial view (final)", perf_counter() - stage_start))

        def step_apply_prepared_cache() -> None:
            prepared_state = state["prepared_review_cache_state"]
            if prepared_state is not None:
                self.apply_prepared_review_cache_state(prepared_state, refresh_ui=False)

        def step_render_cached_dvh() -> None:
            if self.rtstruct is not None and self.ct is not None and bool(state["cache_loaded"]):
                self.render_dvh_plot()
                if self.dvh_curves:
                    self.dvh_status_label.setText("Loaded saved DVH/constraints cache.")
                else:
                    self.dvh_status_label.setText("Saved DVH cache contained no curves.")
                self.update_dvh_cache_button()
                if bool(state["used_preloaded_review_cache"]):
                    self.show_progress_status("Loaded preloaded saved JSON cache")
                else:
                    self.show_progress_status("Loaded saved JSON cache")

        def step_schedule_orthogonal_refresh() -> None:
            if self.ct is not None:
                self.schedule_deferred_patient_view_refresh(
                    timing_entries=timing_entries,
                    folder=folder,
                )

        def step_record_timing() -> None:
            timing_entries.append(("Load saved DVH cache", state["cache_load_duration"]))
            if not bool(state["cache_loaded"]):
                cache_info = state["cache_info"]
                timing_entries.append(("Compute DVH (background)" if cache_info.dvh_can_start else "Compute DVH", None))
            timing_entries.append(("Total patient load to interactive review", perf_counter() - overall_start))

        def step_finish() -> None:
            self._finish_patient_activation_success(
                activation_token=activation_token,
                folder=folder,
                patient_list_index=patient_list_index,
                constraints_path=constraints_path,
                rtstruct_path=rtstruct_path,
                rtdose_paths=rtdose_paths,
                timing_entries=timing_entries,
                cache_info=state["cache_info"],
                cache_loaded=bool(state["cache_loaded"]),
            )

        return self._run_patient_activation_step_sequence(
            activation_token=activation_token,
            folder=folder,
            steps=[
                step_prepare_status,
                step_apply_prepared_cache,
                step_populate_structures,
                step_refresh_axial_final,
                step_render_cached_dvh,
                step_schedule_orthogonal_refresh,
                step_record_timing,
                step_finish,
            ],
            constraints_path=constraints_path,
            rtstruct_path=rtstruct_path,
            rtdose_paths=rtdose_paths,
            timing_entries=timing_entries,
            overall_start=overall_start,
        )

    def _complete_patient_folder_activation(
        self,
        *,
        activation_token: int,
        folder: str,
        patient_list_index: Optional[int],
        payload: PatientPreloadPayload,
        constraints_path: Optional[str],
        rtstruct_path: Optional[str],
        rtdose_paths: List[str],
        derived_array_cache_path: Optional[Path],
        timing_entries: List[Tuple[str, Optional[float]]],
        overall_start: float,
        fast_activate: bool,
    ) -> bool:
        if not self._is_current_patient_activation(activation_token, folder):
            return False

        try:
            cache_info, cache_loaded, cache_load_duration, used_preloaded_review_cache = self._load_saved_review_cache_if_available(
                derived_array_cache_path=derived_array_cache_path,
                preloaded_review_cache_data=payload.review_cache_data,
            )
            self._finalize_patient_load_interactive_state(
                cache_info=cache_info,
                cache_loaded=cache_loaded,
                cache_load_duration=cache_load_duration,
                used_preloaded_review_cache=used_preloaded_review_cache,
                fast_activate=fast_activate,
                patient_folder=folder,
                timing_entries=timing_entries,
                overall_start=overall_start,
            )
            return self._finish_patient_activation_success(
                activation_token=activation_token,
                folder=folder,
                patient_list_index=patient_list_index,
                constraints_path=constraints_path,
                rtstruct_path=rtstruct_path,
                rtdose_paths=rtdose_paths,
                timing_entries=timing_entries,
                cache_info=cache_info,
                cache_loaded=cache_loaded,
            )
        except Exception as e:
            return self._handle_patient_activation_failure(
                activation_token=activation_token,
                folder=folder,
                constraints_path=constraints_path,
                rtstruct_path=rtstruct_path,
                rtdose_paths=rtdose_paths,
                timing_entries=timing_entries,
                overall_start=overall_start,
                error=e,
            )
        finally:
            self.set_patient_activation_ui_locked(False)
            self.set_heavy_view_updates_enabled(True)

    def on_load_patient_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Patient Folder")
        if not folder:
            return
        self.clear_patient_queue_state()
        self.load_patient_folder_path(folder)

    def on_reset_view(self):
        if not self.apply_image_based_view_ranges():
            self.axial_view.autoRange()
            self.sagittal_view.autoRange()
            self.coronal_view.autoRange()
        self.fit_dvh_view_to_visible_curves()

    def on_clear_patient_session(self):
        self.clear_patient_queue_state()
        self.clear_patient_session_state()

    def ensure_structure_filter_dialog(self) -> QtWidgets.QDialog:
        if self.structure_filter_dialog is not None and self.structure_filter_tree_widget is not None:
            return self.structure_filter_dialog

        dialog = QtWidgets.QDialog(self, QtCore.Qt.WindowType.Popup)
        dialog.setWindowTitle("Structures")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hint_label = QtWidgets.QLabel(
            "Use Exclude to hide a structure from the Axial and DVH lists. "
            "Use Target to include a structure as a target subvolume in the Targets tab."
        )
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        tree_widget = QtWidgets.QTreeWidget()
        tree_widget.setColumnCount(3)
        tree_widget.setHeaderLabels(["Exclude", "Target", "Structure"])
        tree_widget.setRootIsDecorated(False)
        tree_widget.setItemsExpandable(False)
        tree_widget.setUniformRowHeights(True)
        tree_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        tree_widget.setMinimumWidth(420)
        tree_widget.setMinimumHeight(340)
        tree_widget.header().setStretchLastSection(True)
        tree_widget.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        tree_widget.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        tree_widget.header().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        tree_widget.itemChanged.connect(self.on_structure_filter_item_changed)
        layout.addWidget(tree_widget)

        self.structure_filter_dialog = dialog
        self.structure_filter_tree_widget = tree_widget
        return dialog

    def populate_structure_filter_dialog(self) -> None:
        if self.structure_filter_tree_widget is None:
            return

        entries = self.get_filterable_structure_entries()
        blocker = QtCore.QSignalBlocker(self.structure_filter_tree_widget)
        self.structure_filter_tree_widget.clear()
        for normalized_name, structure_name in entries:
            item = QtWidgets.QTreeWidgetItem(self.structure_filter_tree_widget)
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, normalized_name)
            item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setText(2, structure_name)
            item.setCheckState(
                0,
                QtCore.Qt.CheckState.Checked
                if normalized_name in self.hidden_structure_names
                else QtCore.Qt.CheckState.Unchecked,
            )
            item.setCheckState(
                1,
                QtCore.Qt.CheckState.Checked
                if normalized_name in self.additional_target_subvolume_names
                else QtCore.Qt.CheckState.Unchecked,
            )
            if normalized_name.startswith("PTV"):
                item.setText(1, "Primary")
                item.setCheckState(1, QtCore.Qt.CheckState.Unchecked)
        del blocker

    def apply_structure_filter_settings(self, *, refresh_lists: bool) -> None:
        if self.rtstruct is None:
            return
        self.target_containment_cache = {}
        self.stereotactic_metrics_cache = {}
        self.stereotactic_volume_context_cache = {}
        self.cached_target_table_rows = None
        if refresh_lists:
            self.populate_structures_list()
            self.on_structure_visibility_changed()
            self.on_dvh_structure_visibility_changed()
        self.update_targets_table()
        self.update_dvh_cache_button()

    def on_structure_filter_item_changed(
        self,
        item: QtWidgets.QTreeWidgetItem,
        column: int,
    ) -> None:
        normalized_name = normalize_structure_name(str(item.data(0, QtCore.Qt.ItemDataRole.UserRole) or ""))
        if not normalized_name:
            return
        hidden_before = normalized_name in self.hidden_structure_names
        target_before = normalized_name in self.additional_target_subvolume_names
        if item.checkState(0) == QtCore.Qt.CheckState.Checked:
            self.hidden_structure_names.add(normalized_name)
        else:
            self.hidden_structure_names.discard(normalized_name)
        if normalized_name.startswith("PTV"):
            if item.checkState(1) != QtCore.Qt.CheckState.Unchecked:
                blocker = QtCore.QSignalBlocker(self.structure_filter_tree_widget)
                item.setCheckState(1, QtCore.Qt.CheckState.Unchecked)
                del blocker
            self.additional_target_subvolume_names.discard(normalized_name)
        elif item.checkState(1) == QtCore.Qt.CheckState.Checked:
            self.additional_target_subvolume_names.add(normalized_name)
        else:
            self.additional_target_subvolume_names.discard(normalized_name)
        hidden_changed = hidden_before != (normalized_name in self.hidden_structure_names)
        target_changed = target_before != (normalized_name in self.additional_target_subvolume_names)
        if hidden_changed or target_changed:
            self.apply_structure_filter_settings(refresh_lists=hidden_changed)

    def on_show_structure_filter_popup(self) -> None:
        if self.rtstruct is None:
            return
        dialog = self.ensure_structure_filter_dialog()
        self.populate_structure_filter_dialog()
        anchor_widget = None
        if hasattr(self, "main_toolbar"):
            anchor_widget = self.main_toolbar.widgetForAction(self.structure_filter_action)
        if anchor_widget is not None:
            popup_origin = anchor_widget.mapToGlobal(
                QtCore.QPoint(0, anchor_widget.height())
            )
        else:
            popup_origin = self.mapToGlobal(QtCore.QPoint(0, 0))
        dialog.move(popup_origin)
        dialog.show()
        dialog.raise_()

    def on_reset_window_level(self):
        self.window_level_slider.setWindowLevel(400, 40)

    def on_tab_changed(self, _index: int):
        self.update_constraints_table_column_widths()
        self.update_targets_table_column_widths()
        if self.constraints_table_refresh_pending and self.tabs.currentWidget() is self.constraints_tab:
            QtCore.QTimer.singleShot(0, self.update_constraints_table)
        if self.targets_table_refresh_pending and self.tabs.currentWidget() is self.targets_tab:
            QtCore.QTimer.singleShot(0, self.update_targets_table)
        if self.dvh_plot_refresh_pending and self.tabs.currentWidget() is self.dvh_tab:
            QtCore.QTimer.singleShot(0, self.render_dvh_plot)

    def on_autoscroll_button_pressed(self):
        if self.autoscroll_button.isChecked():
            self.autoscroll_timer.stop()

    def toggle_autoscroll_shortcut(self):
        if not self.autoscroll_button.isEnabled():
            return
        self.autoscroll_button.click()

    def step_slice(self, delta: int):
        if self.ct is None:
            return
        next_value = int(
            np.clip(
                self.slice_slider.value() + delta,
                self.slice_slider.minimum(),
                self.slice_slider.maximum(),
            )
        )
        self.slice_slider.setValue(next_value)

    def on_previous_slice(self):
        self.step_slice(-1)

    def on_next_slice(self):
        self.step_slice(1)

    def get_autoscroll_slice_step_mm(self) -> Optional[float]:
        if self.ct is None:
            return None
        z_positions = np.asarray(self.ct.z_positions_mm, dtype=float)
        if z_positions.size < 2:
            spacing_z = float(self.ct.spacing_xyz_mm[2])
            return spacing_z if np.isfinite(spacing_z) and spacing_z > 0.0 else None
        slice_steps_mm = np.abs(np.diff(z_positions))
        slice_steps_mm = slice_steps_mm[np.isfinite(slice_steps_mm) & (slice_steps_mm > 0.0)]
        if slice_steps_mm.size == 0:
            spacing_z = float(self.ct.spacing_xyz_mm[2])
            return spacing_z if np.isfinite(spacing_z) and spacing_z > 0.0 else None
        return float(np.median(slice_steps_mm))

    def set_autoscroll_speed_mm_per_s(self, target_speed_mm_per_s: float) -> None:
        slice_step_mm = self.get_autoscroll_slice_step_mm()
        if slice_step_mm is None or target_speed_mm_per_s <= 0.0:
            return
        target_interval_ms = int(round((slice_step_mm / float(target_speed_mm_per_s)) * 1000.0))
        target_interval_ms = int(
            np.clip(
                target_interval_ms,
                self.autoscroll_interval_min_ms,
                self.autoscroll_interval_max_ms,
            )
        )
        self.autoscroll_timer.setInterval(target_interval_ms)
        self.update_autoscroll_speed_label()
        if self.autoscroll_button.isChecked():
            self.autoscroll_timer.start()

    def reset_autoscroll_speed(self) -> None:
        self.set_autoscroll_speed_mm_per_s(self.autoscroll_default_speed_mm_per_s)

    def adjust_autoscroll_speed(self, delta_mm_per_s: float):
        current_speed_mm_per_s = self.get_autoscroll_speed_mm_per_s()
        if current_speed_mm_per_s is None:
            self.reset_autoscroll_speed()
            return
        target_speed_mm_per_s = max(0.1, current_speed_mm_per_s + float(delta_mm_per_s))
        self.set_autoscroll_speed_mm_per_s(target_speed_mm_per_s)

    def on_autoscroll_slower(self):
        self.adjust_autoscroll_speed(-self.autoscroll_speed_step_mm_per_s)

    def on_autoscroll_faster(self):
        self.adjust_autoscroll_speed(self.autoscroll_speed_step_mm_per_s)

    def go_to_point(self, k: int, r: int, c: int):
        self.max_dose_index_zyx = (k, r, c)
        self.current_row = r
        self.current_col = c

        if self.slice_slider.value() != k:
            self.slice_slider.setValue(k)
        else:
            self.update_display()
        self.refresh_orthogonal_views_from_controls()
        self.center_views_on_max_dose()

    def on_go_to_max_dose(self):
        if self.sampled_dose_volume_ct is None:
            return

        if not np.isfinite(self.sampled_dose_volume_ct).any():
            return

        max_index = np.unravel_index(int(np.nanargmax(self.sampled_dose_volume_ct)), self.sampled_dose_volume_ct.shape)
        k, r, c = (int(max_index[0]), int(max_index[1]), int(max_index[2]))
        self.go_to_point(k, r, c)

    def on_go_to_max_tissue(self):
        if self.max_tissue_index_zyx is None:
            self.get_max_tissue_dose_goal_lines()
        if self.max_tissue_index_zyx is None:
            return
        k, r, c = self.max_tissue_index_zyx
        self.go_to_point(k, r, c)

    def on_toggle_dose_overlay(self, checked: bool):
        self.dose_overlay_enabled = checked
        self.dose_toggle_button.setText("On" if checked else "Off")
        self.refresh_all_views()

    def set_view_interaction_enabled(self, enabled: bool):
        for view in (self.axial_view, self.sagittal_view, self.coronal_view):
            view.setMouseEnabled(x=enabled, y=enabled)
            view.setMenuEnabled(enabled)

    def center_view_on_point(self, view: pg.ViewBox, x: float, y: float):
        x_range, y_range = view.viewRange()
        width = max(float(x_range[1] - x_range[0]), 1.0)
        height = max(float(y_range[1] - y_range[0]), 1.0)
        view.setRange(
            xRange=(x - width / 2.0, x + width / 2.0),
            yRange=(y - height / 2.0, y + height / 2.0),
            padding=0.0,
        )

    def center_views_on_max_dose(self):
        center_points = build_max_dose_center_points(self.ct, self.max_dose_index_zyx)
        if center_points is None:
            return
        self.center_view_on_point(self.axial_view, *center_points.axial_point)
        self.center_view_on_point(self.sagittal_view, *center_points.sagittal_point)
        self.center_view_on_point(self.coronal_view, *center_points.coronal_point)

    def set_autoscroll_ui_locked(self, locked: bool):
        self.set_view_interaction_enabled(not locked)
        self.autoscroll_button.setEnabled(True)
        self.autoscroll_slower_button.setEnabled(True)
        self.autoscroll_faster_button.setEnabled(True)
        self.axial_structure_list.set_enabled(not locked)
        self.dvh_structure_list.set_enabled(not locked)
        self.clear_dvh_structures_button.setEnabled(not locked)
        self.slice_prev_button.setEnabled(not locked)
        self.slice_slider.setEnabled(not locked)
        self.slice_next_button.setEnabled(not locked)
        self.window_level_slider.setEnabled(not locked)
        self.reset_window_level_button.setEnabled(not locked)
        self.dose_opacity_slider.setEnabled(not locked and self.dose is not None)
        self.dose_toggle_button.setEnabled(not locked and self.dose is not None)
        self.max_dose_button.setEnabled(not locked and self.sampled_dose_volume_ct is not None)
        self.dose_range_slider.setEnabled(not locked and self.dose is not None)
        self.dose_min_edit.setEnabled(not locked and self.dose is not None)
        self.dose_max_edit.setEnabled(not locked and self.dose is not None)
        for edit in self.isodose_edit_widgets:
            edit.setEnabled(not locked)
        self.load_patient_action.setEnabled(not locked)
        self.load_patient_list_action.setEnabled(not locked)
        self.patient_list_combo.setEnabled((not locked) and bool(self.patient_list_folders))
        self.next_patient_button.setEnabled((not locked) and self.preload_matches_next_patient())
        self.reset_view_action.setEnabled(not locked)
        self.tabs.tabBar().setEnabled(not locked)

    def set_patient_activation_ui_locked(self, locked: bool) -> None:
        if self.patient_activation_ui_locked == locked:
            return
        self.patient_activation_ui_locked = locked

        self.set_view_interaction_enabled(not locked)
        self.autoscroll_button.setEnabled(not locked)
        self.autoscroll_slower_button.setEnabled(not locked)
        self.autoscroll_faster_button.setEnabled(not locked)
        self.axial_structure_list.set_enabled(not locked)
        self.dvh_structure_list.set_enabled(not locked)
        self.clear_dvh_structures_button.setEnabled(not locked)
        self.slice_prev_button.setEnabled(not locked)
        self.slice_slider.setEnabled(not locked)
        self.slice_next_button.setEnabled(not locked)
        self.window_level_slider.setEnabled(not locked)
        self.reset_window_level_button.setEnabled(not locked)
        self.dose_opacity_slider.setEnabled((not locked) and self.dose is not None)
        self.dose_toggle_button.setEnabled((not locked) and self.dose is not None)
        self.max_dose_button.setEnabled((not locked) and self.sampled_dose_volume_ct is not None)
        self.dose_range_slider.setEnabled((not locked) and self.dose is not None)
        self.dose_min_edit.setEnabled((not locked) and self.dose is not None)
        self.dose_max_edit.setEnabled((not locked) and self.dose is not None)
        for edit in self.isodose_edit_widgets:
            edit.setEnabled(not locked)
        self.load_patient_action.setEnabled(not locked)
        self.load_patient_list_action.setEnabled(not locked)
        self.patient_list_combo.setEnabled((not locked) and bool(self.patient_list_folders))
        self.next_patient_button.setEnabled((not locked) and self.preload_matches_next_patient())
        self.reset_view_action.setEnabled(not locked)
        self.tabs.tabBar().setEnabled(not locked)
        self.constraint_sheet_combo.setEnabled((not locked) and bool(self.available_constraint_sheet_names))
        self.add_constraint_button.setEnabled((not locked) and self.rtstruct is not None)
        self.constraints_table.setEnabled(not locked)
        self.targets_table.setEnabled(not locked)
        self.save_cache_action.setEnabled((not locked) and self.current_patient_folder is not None and bool(self.dvh_curves))
        self.print_report_action.setEnabled((not locked) and self.current_patient_folder is not None and self.rtstruct is not None)
        self.structure_filter_action.setEnabled((not locked) and self.rtstruct is not None and bool(self.get_filterable_structure_entries()))

        app = QtWidgets.QApplication.instance()
        if app is not None:
            if locked:
                if app.overrideCursor() is None:
                    app.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            elif app.overrideCursor() is not None:
                app.restoreOverrideCursor()
        if not locked and self.autoscroll_button.isChecked():
            self.set_autoscroll_ui_locked(True)

    def finish_autoscroll_ui(self, clear_checked_state: bool):
        self.autoscroll_timer.stop()
        if clear_checked_state:
            blocker = QtCore.QSignalBlocker(self.autoscroll_button)
            self.autoscroll_button.setChecked(False)
            del blocker
        self.set_autoscroll_ui_locked(False)
        self.autoscroll_button.setText("Autoscroll")

    def reset_autoscroll_ui(self):
        self.finish_autoscroll_ui(clear_checked_state=False)

    def cancel_autoscroll(self):
        self.finish_autoscroll_ui(clear_checked_state=True)

    def set_view_range_from_bounds(
        self,
        view: pg.ViewBox,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        padding_fraction: float = 0.05,
    ):
        width = max(x_max - x_min, 1.0)
        height = max(y_max - y_min, 1.0)
        x_pad = width * padding_fraction / 2.0
        y_pad = height * padding_fraction / 2.0
        view.setRange(
            xRange=(x_min - x_pad, x_max + x_pad),
            yRange=(y_min - y_pad, y_max + y_pad),
            padding=0.0,
        )

    def apply_image_based_view_ranges(self) -> bool:
        if self.ct is None:
            return False

        if self.image_view_bounds is None:
            return False

        k = int(np.clip(self.slice_slider.value(), 0, self.ct.volume_hu.shape[0] - 1))
        axial_bounds = self.image_view_bounds.axial_by_slice.get(k)
        if axial_bounds is not None:
            self.set_view_range_from_bounds(
                self.axial_view,
                axial_bounds[0],
                axial_bounds[1],
                axial_bounds[2],
                axial_bounds[3],
            )

        if self.image_view_bounds.sagittal is not None:
            self.set_view_range_from_bounds(
                self.sagittal_view,
                self.image_view_bounds.sagittal[0],
                self.image_view_bounds.sagittal[1],
                self.image_view_bounds.sagittal[2],
                self.image_view_bounds.sagittal[3],
            )

        if self.image_view_bounds.coronal is not None:
            self.set_view_range_from_bounds(
                self.coronal_view,
                self.image_view_bounds.coronal[0],
                self.image_view_bounds.coronal[1],
                self.image_view_bounds.coronal[2],
                self.image_view_bounds.coronal[3],
            )

        return bool(
            axial_bounds is not None
            or self.image_view_bounds.sagittal is not None
            or self.image_view_bounds.coronal is not None
        )

    def reload_rtstruct_from_current_selection(
        self,
        refresh_dvh: bool = True,
        refresh_views: bool = True,
        refresh_lists: bool = True,
    ):
        if self.ct is None or not self.rtstruct_path:
            return

        self.rtstruct = load_rtstruct(
            self.rtstruct_path,
            self.ct,
        )
        self.sort_rtstruct_structures_for_display()
        self.structure_mask_cache = None
        self.structure_mask_cache_names = []
        self.target_curve_cache = {}
        self.target_metrics_cache = {}
        self.stereotactic_metrics_cache = {}
        self.stereotactic_volume_context_cache = {}
        self.max_tissue_dose_gy_cache = None
        self.max_tissue_index_zyx = None
        self.ptv_union_slice_mask_cache = None
        self.ptv_union_volume_mask_cache = None
        self.target_slice_mask_cache = {}
        self.target_containment_cache = {}
        self.structure_volume_mask_cache = {}
        self.structure_geometry_volume_cache = {}
        self.cached_target_table_rows = None
        if refresh_lists:
            self.populate_structures_list()
            self.populate_isodose_controls()
        if refresh_dvh:
            self.refresh_dvh()
        if refresh_views:
            self.refresh_all_views()

    def sort_rtstruct_structures_for_display(self):
        if self.rtstruct is None:
            return

        csv_order = {name: idx for idx, name in enumerate(self.structure_csv_order)}
        indexed_structures = list(enumerate(self.rtstruct.structures))
        indexed_structures.sort(
            key=lambda entry: (
                0 if normalize_structure_name(entry[1].name).startswith("PTV") else 1,
                csv_order.get(normalize_structure_name(entry[1].name), 10**6),
                entry[0],
            )
        )
        self.rtstruct.structures = [structure for _, structure in indexed_structures]

    def write_load_timing_report(
        self,
        folder: str,
        timing_entries: List[Tuple[str, Optional[float]]],
        csv_path: Optional[str],
        rtstruct_path: Optional[str],
        rtdose_paths: List[str],
        error_message: Optional[str] = None,
    ) -> Optional[Path]:
        report_path = Path(folder) / "peer_load_timing.txt"
        try:
            report_path.write_text(
                build_load_timing_report_text(
                    folder=folder,
                    timing_entries=timing_entries,
                    constraints_path=csv_path,
                    constraints_sheet_name=self.constraints_sheet_name,
                    rtstruct_path=rtstruct_path,
                    rtdose_paths=rtdose_paths,
                    ct=self.ct,
                    rtstruct=self.rtstruct,
                    error_message=error_message,
                ),
                encoding="utf-8",
            )
        except OSError:
            return None

        return report_path

    def get_dvh_cache_path(self) -> Optional[Path]:
        return compute_dvh_cache_path(self.current_patient_folder)

    def get_derived_array_cache_path(self, base_path: Optional[Path] = None) -> Optional[Path]:
        return compute_derived_array_cache_path(base_path if base_path is not None else self.get_dvh_cache_path())

    def get_ct_geometry_signature(self) -> Optional[str]:
        return compute_ct_geometry_signature(self.ct)

    def get_derived_array_cache_signature(self) -> Dict[str, str]:
        return compute_derived_array_cache_signature(
            sample_dose_to_ct_slice_func=sample_dose_to_ct_slice,
            build_structure_slice_mask_func=build_structure_slice_mask,
            get_target_structure_slice_masks_func=type(self).get_target_structure_slice_masks,
            get_ptv_union_slice_masks_func=type(self).get_ptv_union_slice_masks,
            load_rtstruct_func=load_rtstruct,
        )

    def get_derived_cache_structures(self) -> List[StructureSliceContours]:
        return select_derived_cache_structures(self.rtstruct, self.additional_target_subvolume_names)

    def update_dvh_cache_button(self):
        self.save_cache_action.setEnabled(
            self.current_patient_folder is not None and bool(self.dvh_curves)
        )
        self.print_report_action.setEnabled(
            self.current_patient_folder is not None and self.rtstruct is not None
        )
        self.structure_filter_action.setEnabled(self.rtstruct is not None and bool(self.get_filterable_structure_entries()))
        self.add_constraint_button.setEnabled(self.rtstruct is not None)

    def _refresh_sidebar_label_geometry(self, label: QtWidgets.QLabel) -> None:
        label.ensurePolished()
        if not label.text().strip():
            label.setFixedHeight(0)
            return
        height = max(
            label.sizeHint().height(),
            QtGui.QFontMetrics(label.font()).lineSpacing() * max(1, len(label.text().splitlines()))
            + (2 * label.margin())
            + 2,
        )
        label.setFixedHeight(height)
        label.updateGeometry()

    def refresh_patient_plan_label_layout(self, *, pump_events: bool = False) -> None:
        self._refresh_sidebar_label_geometry(self.patient_name_label)
        self._refresh_sidebar_label_geometry(self.patient_plan_label)
        if hasattr(self, "patient_summary_widget") and self.patient_summary_widget is not None:
            self.patient_summary_widget.layout().activate()
            self.patient_summary_widget.updateGeometry()
        if hasattr(self, "tabs") and self.tabs is not None:
            self.tabs.updateGeometry()
            self.tabs.repaint()
            self.tabs.tabBar().updateGeometry()
            self.tabs.tabBar().repaint()
        if hasattr(self, "patient_summary_widget") and self.patient_summary_widget is not None:
            self.patient_summary_widget.repaint()
        self.patient_name_label.repaint()
        self.patient_plan_label.repaint()
        if hasattr(self, "patient_summary_widget") and self.patient_summary_widget is not None:
            QtCore.QCoreApplication.sendPostedEvents(self.patient_summary_widget, int(QtCore.QEvent.Type.Paint))
        QtCore.QCoreApplication.sendPostedEvents(self.patient_name_label, int(QtCore.QEvent.Type.Paint))
        QtCore.QCoreApplication.sendPostedEvents(self.patient_plan_label, int(QtCore.QEvent.Type.Paint))
        if hasattr(self, "tabs") and self.tabs is not None:
            QtCore.QCoreApplication.sendPostedEvents(self.tabs, int(QtCore.QEvent.Type.Paint))
            QtCore.QCoreApplication.sendPostedEvents(self.tabs.tabBar(), int(QtCore.QEvent.Type.Paint))
        if pump_events:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def update_patient_plan_label(self, *, pump_events: bool = False):
        if not self.patient_plan_lines:
            self.patient_name_label.clear()
            self.patient_name_label.setFixedHeight(0)
            self.patient_name_label.setVisible(False)
            self.patient_plan_label.clear()
            self.patient_plan_label.setFixedHeight(0)
            self.patient_plan_label.setVisible(False)
            self.patient_summary_widget.setVisible(False)
            return

        patient_name = str(self.patient_plan_lines[0]).strip()
        detail_parts = [
            str(line).strip()
            for line in self.patient_plan_lines[1:]
            if str(line).strip()
        ]
        if not detail_parts:
            details_text = ""
        elif len(detail_parts) == 1:
            details_text = detail_parts[0]
        else:
            details_text = detail_parts[0] + (" " * 10) + " | ".join(detail_parts[1:])
        self.patient_name_label.setText(patient_name)
        self.patient_name_label.setVisible(bool(patient_name))
        self.patient_plan_label.setText(details_text)
        self.patient_plan_label.setVisible(bool(details_text))
        self.patient_summary_widget.setVisible(bool(patient_name or details_text))
        self.refresh_patient_plan_label_layout(pump_events=pump_events)

    def set_patient_plan_lines(
        self,
        patient_plan_lines: Optional[Sequence[str]],
        *,
        pump_events: bool = False,
    ) -> None:
        self.patient_plan_lines = tuple(patient_plan_lines) if patient_plan_lines else None
        self.update_patient_plan_label(pump_events=pump_events)

    def try_stage_patient_plan_lines_from_cache(self, folder: str, *, pump_events: bool = False) -> bool:
        cache_path = compute_dvh_cache_path(folder)
        if cache_path is None or not cache_path.exists():
            return False
        loaded_cache = load_review_cache_file(cache_path)
        if loaded_cache is None:
            return False
        patient_plan_lines_payload = loaded_cache.payload.get("patient_plan_lines")
        if not isinstance(patient_plan_lines_payload, list):
            return False
        patient_plan_lines = [str(line).strip() for line in patient_plan_lines_payload if str(line).strip()]
        if not patient_plan_lines:
            return False
        self.set_patient_plan_lines(patient_plan_lines, pump_events=pump_events)
        return True

    def refresh_constraint_sheet_combo(self, preferred_sheet_name: Optional[str] = None) -> None:
        workbook_path = get_constraints_workbook_path()
        self.structure_filter_csv_path = workbook_path
        self.constraint_workbook_error = None
        if workbook_path is None:
            self.available_constraint_sheet_names = []
            self.constraints_sheet_name = None
            blocker = QtCore.QSignalBlocker(self.constraint_sheet_combo)
            self.constraint_sheet_combo.clear()
            self.constraint_sheet_combo.setEnabled(False)
            del blocker
            return

        try:
            sheet_names = list_constraints_workbook_sheets(workbook_path)
        except Exception as exc:
            logger.warning("Failed to read constraints workbook: %s (%s)", workbook_path, exc)
            self.constraint_workbook_error = str(exc)
            sheet_names = []

        self.available_constraint_sheet_names = sheet_names
        if preferred_sheet_name in {None, NO_CONSTRAINTS_SHEET_LABEL}:
            selected_sheet_name = None
        elif preferred_sheet_name in sheet_names:
            selected_sheet_name = preferred_sheet_name
        elif self.constraints_sheet_name in sheet_names:
            selected_sheet_name = self.constraints_sheet_name
        else:
            selected_sheet_name = None
        self.constraints_sheet_name = selected_sheet_name

        blocker = QtCore.QSignalBlocker(self.constraint_sheet_combo)
        self.constraint_sheet_combo.clear()
        self.constraint_sheet_combo.addItem(NO_CONSTRAINTS_SHEET_LABEL)
        self.constraint_sheet_combo.addItems(sheet_names)
        if selected_sheet_name is not None:
            self.constraint_sheet_combo.setCurrentText(selected_sheet_name)
        else:
            self.constraint_sheet_combo.setCurrentText(NO_CONSTRAINTS_SHEET_LABEL)
        self.constraint_sheet_combo.setEnabled(bool(sheet_names))
        del blocker
        if self.constraint_workbook_error:
            self.statusBar().showMessage(
                f"Could not read constraints workbook: {self.constraint_workbook_error}",
                8000,
            )

    def apply_constraint_sheet(
        self,
        sheet_name: Optional[str],
        *,
        refresh_lists: bool = True,
        refresh_dvh: bool = True,
    ) -> None:
        workbook_path = get_constraints_workbook_path()
        self.structure_filter_csv_path = workbook_path
        self.constraints_sheet_name = sheet_name if sheet_name else None

        if workbook_path is None or not sheet_name:
            self.csv_structure_goals_by_name = {}
            self.structure_csv_order = []
        else:
            _, self.csv_structure_goals_by_name, self.structure_csv_order = load_structure_constraints_sheet(
                workbook_path,
                sheet_name,
                self.plan_phases,
            )
        self.rebuild_structure_goals_by_name()
        self.cached_target_table_rows = None

        if self.rtstruct is None:
            self.update_targets_table()
            return

        self.sort_rtstruct_structures_for_display()
        if refresh_lists:
            self.populate_structures_list()
            newly_required_names = [
                normalized_name
                for normalized_name in self.csv_structure_goals_by_name
                if self.is_listable_structure_name(normalized_name)
            ]
            selection_changed = False
            for normalized_name in newly_required_names:
                if self.dvh_structure_list.set_checked(normalized_name, True):
                    selection_changed = True

            current_curve_names = set(self.current_dvh_curve_names())
            if refresh_dvh and (
                selection_changed
                or any(name not in current_curve_names for name in newly_required_names)
            ):
                self.refresh_dvh()
            else:
                self.refresh_visible_structure_goal_evaluations()
                self.update_dvh_goal_evaluation_cache()
                self.update_structure_list_goal_texts()
        self.update_targets_table()

    def on_constraint_sheet_changed(self, sheet_name: str) -> None:
        normalized_sheet_name = sheet_name.strip()
        if not normalized_sheet_name:
            return
        target_sheet_name = None if normalized_sheet_name == NO_CONSTRAINTS_SHEET_LABEL else normalized_sheet_name
        if target_sheet_name == self.constraints_sheet_name and (
            self.csv_structure_goals_by_name or target_sheet_name is None
        ):
            return
        self.apply_constraint_sheet(target_sheet_name, refresh_lists=True, refresh_dvh=True)

    def serialize_dvh_curve(self, curve: DVHCurve) -> Dict[str, object]:
        return serialize_dvh_curve_payload(curve)

    def deserialize_dvh_curve(self, payload: Dict[str, object]) -> DVHCurve:
        return deserialize_dvh_curve_payload(payload)

    def serialize_goal_evaluations(self) -> Dict[str, List[Dict[str, object]]]:
        return serialize_goal_evaluations_payload(self.structure_goal_evaluations)

    def deserialize_goal_evaluations(
        self,
        payload: object,
    ) -> Optional[Dict[str, List[StructureGoalEvaluation]]]:
        return deserialize_goal_evaluations_payload(payload)

    def serialize_structure_goals(
        self,
        goals_by_structure: Dict[str, List[StructureGoal]],
    ) -> Dict[str, List[Dict[str, str]]]:
        return serialize_structure_goals_payload(goals_by_structure)

    def deserialize_structure_goals(
        self,
        payload: object,
    ) -> Optional[Dict[str, List[StructureGoal]]]:
        return deserialize_structure_goals_payload(payload)

    def rebuild_structure_goals_by_name(self) -> None:
        merged: Dict[str, List[StructureGoal]] = {}
        for source in (self.csv_structure_goals_by_name, self.custom_structure_goals_by_name):
            for structure_name, goals in source.items():
                if not goals:
                    continue
                merged.setdefault(structure_name, []).extend(goals)
        self.structure_goals_by_name = merged
        self.dvh_structure_goal_evaluation_cache = {}

    def serialize_target_table_rows(self, rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
        return serialize_target_table_rows_payload(rows)

    def deserialize_target_table_rows(self, payload: object) -> Optional[List[Dict[str, object]]]:
        return deserialize_target_table_rows_payload(payload)

    def get_target_row_reference_dose_text(self, row: Dict[str, object]) -> str:
        return get_target_row_reference_dose_text_helper(
            row,
            normalize_dose_text=self.normalize_stereotactic_dose_text,
        )

    def target_table_rows_require_recompute(self, rows: List[Dict[str, object]]) -> bool:
        return target_table_rows_require_recompute_helper(
            rows,
            has_ct=self.ct is not None,
            has_dose=self.dose is not None,
            stereotactic_summary_enabled=self.stereotactic_summary_enabled(),
        )

    def get_target_method_signature(self) -> Dict[str, object]:
        return {
            "dvh_helper_signature": get_dvh_method_signature(),
            "get_primary_target_context": callable_signature_hash(type(self).get_primary_target_context),
            "build_target_table_rows": callable_signature_hash(type(self).build_target_table_rows),
            "compute_stereotactic_indices": callable_signature_hash(type(self).compute_stereotactic_indices),
            "get_nested_target_structures": callable_signature_hash(type(self).get_nested_target_structures),
            "get_max_tissue_dose_goal_lines": callable_signature_hash(type(self).get_max_tissue_dose_goal_lines),
            "get_default_stereotactic_dose_text": callable_signature_hash(
                type(self).get_default_stereotactic_dose_text
            ),
        }

    def get_constraint_note_key(self, normalized_name: str, goal: StructureGoal) -> str:
        return "||".join(
            [
                normalized_name,
                goal.metric.strip(),
                goal.comparator.strip(),
                goal.value_text.strip(),
            ]
        )

    def get_target_note_key(self, normalized_name: str, parent_normalized_name: Optional[str] = None) -> str:
        if parent_normalized_name:
            return f"TARGET||{parent_normalized_name}||{normalized_name}"
        return f"TARGET||{normalized_name}"

    def get_target_note_key_for_row(self, row: Dict[str, object]) -> str:
        normalized_name = normalize_structure_name(str(row.get("normalized_name", "")))
        parent_normalized_name = row.get("parent_normalized_name")
        is_primary_ptv = bool(row.get("is_primary_ptv", False))
        section_key = (
            normalized_name
            if is_primary_ptv or not parent_normalized_name
            else normalize_structure_name(str(parent_normalized_name))
        )
        if is_primary_ptv:
            return self.get_target_note_key(normalized_name)
        return self.get_target_note_key(normalized_name, parent_normalized_name=section_key)

    def compose_target_note_text(self, computed_note_text: str, stored_note_text: str) -> str:
        return compose_target_note_text_helper(computed_note_text, stored_note_text)

    def build_target_notes_for_save(self, target_rows: List[Dict[str, object]]) -> Dict[str, str]:
        return build_target_notes_for_save_helper(
            target_rows,
            target_notes=self.target_notes,
            get_target_note_key_for_row=self.get_target_note_key_for_row,
        )

    def extract_manual_target_notes(
        self,
        saved_notes_payload: Dict[str, str],
        target_rows: List[Dict[str, object]],
    ) -> Dict[str, str]:
        return extract_manual_target_notes_helper(
            saved_notes_payload,
            target_rows,
            get_target_note_key_for_row=self.get_target_note_key_for_row,
        )

    def get_listable_structure_names(self) -> List[str]:
        if self.rtstruct is None:
            return []
        return [
            structure.name
            for structure in self.rtstruct.structures
            if self.is_base_listable_structure_name(normalize_structure_name(structure.name))
        ]

    def get_filterable_structure_entries(self) -> List[Tuple[str, str]]:
        if self.rtstruct is None:
            return []
        entries: List[Tuple[str, str]] = []
        for structure in self.rtstruct.structures:
            normalized_name = normalize_structure_name(structure.name)
            if not self.is_base_listable_structure_name(normalized_name):
                continue
            entries.append((normalized_name, structure.name))
        return entries

    def parse_constraint_goal_input(self, goal_text: str) -> Optional[Tuple[str, str]]:
        return parse_constraint_goal_input_helper(goal_text)

    def build_custom_constraint_from_editor(self) -> Optional[Tuple[str, StructureGoal]]:
        return build_custom_constraint_from_editor_helper(self.constraint_editor_state)

    def get_constraint_goal_key(self, goal: StructureGoal) -> Tuple[str, str, str]:
        return get_constraint_goal_key_helper(goal)

    def custom_constraint_exists(self, normalized_name: str, goal: StructureGoal) -> bool:
        return custom_constraint_exists_helper(
            normalized_name,
            goal,
            structure_goals_by_name=self.structure_goals_by_name,
        )

    def is_custom_only_constraint(self, normalized_name: str, goal: StructureGoal) -> bool:
        goal_key = self.get_constraint_goal_key(goal)
        in_custom = any(
            self.get_constraint_goal_key(existing_goal) == goal_key
            for existing_goal in self.custom_structure_goals_by_name.get(normalized_name, [])
        )
        if not in_custom:
            return False
        in_csv = any(
            self.get_constraint_goal_key(existing_goal) == goal_key
            for existing_goal in self.csv_structure_goals_by_name.get(normalized_name, [])
        )
        return not in_csv

    def ensure_dvh_structure_selected(self, normalized_name: str) -> bool:
        if not self.is_listable_structure_name(normalized_name):
            return False
        changed = self.dvh_structure_list.set_checked(normalized_name, True)
        if changed:
            self.update_dvh_cache_button()
        return changed

    def on_add_constraint_clicked(self):
        structure_names = self.get_listable_structure_names()
        if not structure_names:
            QtWidgets.QMessageBox.information(
                self,
                "No structures available",
                "Load a patient folder with RTSTRUCT data before adding a constraint.",
            )
            return

        if self.constraint_editor_state is None:
            self.constraint_editor_state = build_initial_constraint_editor_state(structure_names)
            self.update_constraints_table()
        QtCore.QTimer.singleShot(0, self.focus_constraint_editor_metric_edit)

    def focus_constraint_editor_metric_edit(self):
        metric_edit = self.constraint_editor_widgets.get("metric_edit")
        if isinstance(metric_edit, QtWidgets.QLineEdit):
            metric_edit.setFocus()
            metric_edit.selectAll()

    def on_constraint_editor_field_changed(self, field_name: str, value: str):
        if self.constraint_editor_state is None:
            return
        self.constraint_editor_state[field_name] = value
        self.update_constraint_editor_preview()

    def update_constraint_editor_preview(self):
        result_label = self.constraint_editor_widgets.get("result_label")
        add_button = self.constraint_editor_widgets.get("add_button")
        if not isinstance(result_label, QtWidgets.QLabel) or not isinstance(add_button, QtWidgets.QPushButton):
            return

        preview_state = build_constraint_editor_preview_state(
            self.constraint_editor_state,
            structure_goals_by_name=self.structure_goals_by_name,
            get_curve_for_name=self.get_curve_for_name,
            dvh_structure_is_visible=self.dvh_structure_is_visible,
            structure_goal_line_color=self.structure_goal_line_color,
        )
        result_label.setStyleSheet(f"color: {preview_state.color_name};")
        result_label.setText(preview_state.text)
        add_button.setEnabled(preview_state.add_enabled)

    def cancel_constraint_editor(self):
        self.constraint_editor_state = None
        self.constraint_editor_widgets = {}
        self.update_constraints_table()

    def commit_constraint_editor(self):
        built_goal = self.build_custom_constraint_from_editor()
        if built_goal is None:
            self.update_constraint_editor_preview()
            return

        normalized_name, goal = built_goal
        if self.custom_constraint_exists(normalized_name, goal):
            self.cancel_constraint_editor()
            return

        self.custom_structure_goals_by_name.setdefault(normalized_name, []).append(goal)
        self.rebuild_structure_goals_by_name()

        self.constraint_editor_state = None
        self.constraint_editor_widgets = {}

        selected_changed = self.ensure_dvh_structure_selected(normalized_name)
        current_curve_names = set(self.current_dvh_curve_names())
        self.refresh_visible_structure_goal_evaluations()
        self.update_dvh_goal_evaluation_cache()
        self.update_structure_list_goal_texts()
        self.update_dvh_cache_button()

        if selected_changed or normalized_name not in current_curve_names:
            self.refresh_dvh()

    def save_dvh_cache(self, path: Path) -> Optional[str]:
        if self.rtstruct is None:
            raise ValueError("No RTSTRUCT data is loaded.")
        target_rows = self.get_target_table_rows()
        selected_constraint_set = self.constraints_sheet_name or NO_CONSTRAINTS_SHEET_LABEL
        derived_array_cache_path = self.get_derived_array_cache_path(path)
        screenshot_path = self.get_review_screenshot_path(path)
        movie_path = self.get_review_movie_path(path)
        payload = build_review_cache_payload(
            patient_plan_lines=list(self.patient_plan_lines or []),
            selected_constraint_set=selected_constraint_set,
            constraints_file_name=Path(self.structure_filter_csv_path).name if self.structure_filter_csv_path else None,
            constraints_sheet_name=self.constraints_sheet_name,
            rtstruct_file_name=Path(self.rtstruct_path).name if self.rtstruct_path else None,
            constraints_fingerprint=build_file_fingerprint(self.structure_filter_csv_path),
            rtstruct_fingerprint=build_file_fingerprint(self.rtstruct_path),
            rtdose_fingerprints=build_file_fingerprints(self.latest_timing_rtdose_paths),
            rtplan_fingerprints=build_file_fingerprints(self.current_rtplan_paths),
            derived_array_cache_file_name=derived_array_cache_path.name if derived_array_cache_path is not None else None,
            derived_array_cache_signature=self.get_derived_array_cache_signature(),
            structure_names=[normalize_structure_name(structure.name) for structure in self.rtstruct.structures],
            dvh_structure_names=self.get_selected_dvh_structure_names(),
            dvh_mode=self.get_dvh_mode(),
            dvh_method_signature=get_dvh_method_signature(),
            target_method_signature=self.get_target_method_signature(),
            curves=self.dvh_curves,
            custom_constraints=self.custom_structure_goals_by_name,
            goal_evaluations=self.structure_goal_evaluations,
            target_table_rows=target_rows,
            max_tissue_payload=self.build_max_tissue_payload(),
            stereotactic_target_doses=self.stereotactic_target_dose_text_by_name,
            isodose_level_texts=[edit.text().strip() for edit in self.isodose_edit_widgets],
            isodose_colors=self.isodose_colors,
            hidden_structure_names=sorted(self.hidden_structure_names),
            additional_target_subvolume_names=sorted(self.additional_target_subvolume_names),
            constraint_notes=self.constraint_notes,
            target_notes=self.build_target_notes_for_save(target_rows),
        )
        write_json_atomic(path, payload)

        warnings: List[str] = []

        if screenshot_path is not None:
            screenshot_warning = self.save_review_screenshot(screenshot_path)
            if screenshot_warning:
                warnings.append(screenshot_warning)

        if movie_path is not None:
            movie_warning = self.save_review_movie(movie_path)
            if movie_warning:
                warnings.append(movie_warning)

        if derived_array_cache_path is not None:
            try:
                self.save_derived_array_cache(derived_array_cache_path)
            except Exception as exc:
                logger.warning("Saved JSON review cache but failed to save derived array cache: %s", exc)
                warnings.append(f"Failed to save the derived array cache: {exc}")

        if warnings:
            return "Saved review data, but " + " ".join(warnings)
        return None

    def get_review_screenshot_path(self, base_path: Optional[Path] = None) -> Optional[Path]:
        cache_path = base_path if base_path is not None else self.get_dvh_cache_path()
        if cache_path is None:
            return None
        return cache_path.with_name(f"{cache_path.stem}_screenshot.png")

    def save_review_screenshot(self, path: Path) -> Optional[str]:
        self.pump_viewer_ui()
        source_widget = getattr(self, "axial_tab_widget", getattr(self, "viewer_widget", self))
        pixmap = source_widget.grab()
        if pixmap.isNull():
            return "failed to save the review screenshot."
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path_str = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=path.suffix, dir=path.parent)
        temp_path = Path(temp_path_str)
        try:
            try:
                Path(temp_path).unlink()
            except OSError:
                pass
            if not pixmap.save(str(temp_path), "PNG"):
                return "failed to save the review screenshot."
            temp_path.replace(path)
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            logger.warning("Saved JSON review cache but failed to save screenshot: %s", exc)
            return f"failed to save the review screenshot: {exc}"
        return None

    def build_max_tissue_payload(self) -> Optional[Dict[str, object]]:
        if self.ct is None or self.rtstruct is None or self.sampled_dose_volume_ct is None:
            return None

        if self.max_tissue_dose_gy_cache is None:
            previous_defer = self.defer_sidebar_summary_metrics
            self.defer_sidebar_summary_metrics = False
            try:
                self.get_max_tissue_dose_goal_lines()
            finally:
                self.defer_sidebar_summary_metrics = previous_defer

        if self.max_tissue_dose_gy_cache is None:
            return None

        payload: Dict[str, object] = {
            "dose_gy": float(self.max_tissue_dose_gy_cache),
        }
        if self.max_tissue_index_zyx is not None:
            payload["index_zyx"] = [int(value) for value in self.max_tissue_index_zyx]
        return payload

    def save_derived_array_cache(self, path: Path) -> None:
        if self.ct is None or self.rtstruct is None:
            raise ValueError("No CT/RTSTRUCT data is loaded.")
        derived_structures = self.get_derived_cache_structures()
        structure_order = [normalize_structure_name(structure.name) for structure in derived_structures]
        structure_volume_masks = {
            normalize_structure_name(structure.name): self.get_structure_volume_mask(structure)
            for structure in derived_structures
        }
        structure_geometry_volumes_cc = {
            normalize_structure_name(structure.name): float(self.get_structure_geometry_volume_cc(structure))
            for structure in derived_structures
        }
        save_derived_array_cache_file(
            path,
            ct=self.ct,
            rtstruct=self.rtstruct,
            rtstruct_path=self.rtstruct_path,
            rtdose_paths=self.latest_timing_rtdose_paths,
            array_cache_signature=self.get_derived_array_cache_signature(),
            sampled_dose_volume_ct=self.sampled_dose_volume_ct,
            ptv_union_volume_mask=self.get_ptv_union_volume_mask(),
            structure_order=structure_order,
            structure_volume_masks=structure_volume_masks,
            structure_geometry_volumes_cc=structure_geometry_volumes_cc,
        )

    def load_derived_array_cache(self, path: Path) -> bool:
        if self.ct is None:
            return False
        loaded_cache = load_derived_array_cache_file(
            path,
            ct=self.ct,
            rtstruct_path=self.rtstruct_path,
            rtdose_paths=self.latest_timing_rtdose_paths,
            array_cache_signature=self.get_derived_array_cache_signature(),
        )
        if loaded_cache is None:
            return False
        loaded_any = False
        if loaded_cache.rtstruct is not None:
            self.rtstruct = loaded_cache.rtstruct
            self.sort_rtstruct_structures_for_display()
            loaded_any = True
        if loaded_cache.sampled_dose_volume_ct is not None:
            self.sampled_dose_volume_ct = loaded_cache.sampled_dose_volume_ct
            loaded_any = True
        if loaded_cache.ptv_union_volume_mask is not None:
            self.ptv_union_volume_mask_cache = loaded_cache.ptv_union_volume_mask
            self.ptv_union_slice_mask_cache = None
            loaded_any = True
        for normalized_name, cached_mask in loaded_cache.structure_volume_masks.items():
            self.structure_volume_mask_cache[normalized_name] = cached_mask
            self.target_slice_mask_cache.pop(normalized_name, None)
            loaded_any = True
        for normalized_name, geometry_volume_cc in loaded_cache.structure_geometry_volumes_cc.items():
            self.structure_geometry_volume_cache[normalized_name] = geometry_volume_cc
            loaded_any = True
        return loaded_any

    def try_load_derived_array_cache(self) -> bool:
        cache_path = self.get_derived_array_cache_path()
        if cache_path is None or not cache_path.exists():
            return False
        return self.load_derived_array_cache(cache_path)

    def prepare_saved_review_cache_data(
        self,
        loaded_cache: ReviewCacheFileData,
    ) -> Optional[PreparedReviewCacheState]:
        if self.rtstruct is None:
            return None
        return prepare_review_cache_state(
            loaded_cache,
            expected_structure_names=[structure.name for structure in self.rtstruct.structures],
            available_constraint_sheet_names=self.available_constraint_sheet_names,
            no_constraints_sheet_label=NO_CONSTRAINTS_SHEET_LABEL,
            constraints_sheet_name=self.constraints_sheet_name,
            structure_filter_csv_path=self.structure_filter_csv_path,
            rtstruct_path=self.rtstruct_path,
            rtdose_paths=self.latest_timing_rtdose_paths,
            rtplan_paths=self.current_rtplan_paths,
            dvh_mode=self.get_dvh_mode(),
            dvh_method_signature=get_dvh_method_signature(),
            target_method_signature=self.get_target_method_signature(),
            has_ct=self.ct is not None,
            has_dose=self.dose is not None,
            is_base_listable_structure_name=self.is_base_listable_structure_name,
        )

    def apply_prepared_review_cache_state(
        self,
        prepared_state: PreparedReviewCacheState,
        *,
        refresh_ui: bool = True,
    ) -> bool:
        saved_constraints_sheet = prepared_state.selected_constraint_sheet
        if saved_constraints_sheet != self.constraints_sheet_name:
            self.refresh_constraint_sheet_combo(
                preferred_sheet_name=saved_constraints_sheet or NO_CONSTRAINTS_SHEET_LABEL
            )
            self.apply_constraint_sheet(
                saved_constraints_sheet,
                refresh_lists=refresh_ui,
                refresh_dvh=False,
            )

        self.custom_structure_goals_by_name = prepared_state.custom_constraints
        self.rebuild_structure_goals_by_name()
        self.stereotactic_target_dose_text_by_name = dict(prepared_state.stereotactic_target_doses)
        self.apply_isodose_settings(
            level_texts=prepared_state.isodose_level_texts,
            colors=prepared_state.isodose_colors,
        )
        self.hidden_structure_names = set(prepared_state.hidden_structure_names)
        self.additional_target_subvolume_names = set(prepared_state.additional_target_subvolume_names)
        self.constraint_notes = dict(prepared_state.constraint_notes)
        self.target_notes = self.extract_manual_target_notes(
            dict(prepared_state.target_notes_payload),
            prepared_state.target_table_rows,
        )
        self.dvh_curves = list(prepared_state.curves)
        self.max_tissue_dose_gy_cache = prepared_state.max_tissue_dose_gy
        self.max_tissue_index_zyx = prepared_state.max_tissue_index_zyx
        self.structure_mask_cache = None
        self.structure_mask_cache_names = []
        self.dvh_request_structure_names = {}
        self.cached_target_table_rows = prepared_state.cached_target_table_rows
        self.update_dvh_secondary_metric_caches()
        if prepared_state.saved_selected_names:
            if refresh_ui:
                self.dvh_structure_list.set_checked_names(prepared_state.saved_selected_names)
            else:
                self.pending_saved_dvh_selected_names = list(prepared_state.saved_selected_names)
        else:
            self.pending_saved_dvh_selected_names = None
        self.refresh_visible_structure_goal_evaluations(precomputed=prepared_state.goal_evaluations or None)
        self.update_dvh_goal_evaluation_cache(prepared_state.goal_evaluations or None)
        if refresh_ui:
            self.update_structure_list_goal_texts()
            self.render_dvh_plot()
            if self.dvh_curves:
                self.dvh_status_label.setText("Loaded saved DVH/constraints cache.")
            else:
                self.dvh_status_label.setText("Saved DVH cache contained no curves.")
            self.update_dvh_cache_button()
        return True

    def load_saved_review_cache_data(
        self,
        loaded_cache: ReviewCacheFileData,
        *,
        refresh_ui: bool = True,
    ) -> bool:
        prepared_state = self.prepare_saved_review_cache_data(loaded_cache)
        if prepared_state is None:
            return False
        return self.apply_prepared_review_cache_state(prepared_state, refresh_ui=refresh_ui)

    def load_saved_dvh_cache(self, path: Path, *, refresh_ui: bool = True) -> bool:
        loaded_cache = load_review_cache_file(path)
        if loaded_cache is None:
            return False
        return self.load_saved_review_cache_data(loaded_cache, refresh_ui=refresh_ui)

    def try_load_saved_dvh_cache(self, *, refresh_ui: bool = True) -> bool:
        cache_path = self.get_dvh_cache_path()
        if cache_path is None or not cache_path.exists():
            return False
        return self.load_saved_dvh_cache(cache_path, refresh_ui=refresh_ui)

    def on_save_dvh_cache(self):
        cache_path = self.get_dvh_cache_path()
        if cache_path is None:
            QtWidgets.QMessageBox.information(self, "No patient folder", "Load a patient folder before saving DVH data.")
            return
        if not self.dvh_curves:
            QtWidgets.QMessageBox.information(self, "No DVH data", "Wait for DVH results before saving.")
            return
        try:
            save_warning = self.save_dvh_cache(cache_path)
        except (OSError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return
        if save_warning:
            QtWidgets.QMessageBox.warning(self, "Saved with warning", save_warning)
        self.statusBar().showMessage(f"Saved DVH cache to {cache_path.name}", 4000)

    def get_default_report_path(self) -> Path:
        base_dir = Path(self.current_patient_folder) if self.current_patient_folder else Path.home()
        patient_name = ""
        if self.patient_plan_lines:
            patient_name = str(self.patient_plan_lines[0]).strip()
        cleaned_name = re.sub(r"[^A-Za-z0-9._-]+", "_", patient_name).strip("._")
        stem = f"{cleaned_name}_peer_review_report" if cleaned_name else "peer_review_report"
        return base_dir / f"{stem}.pdf"

    def build_constraints_report_rows(self) -> List[Dict[str, object]]:
        if self.rtstruct is None:
            return []

        rows: List[Dict[str, object]] = []
        for structure in self.rtstruct.structures:
            normalized_name = normalize_structure_name(structure.name)
            goals = self.structure_goals_by_name.get(normalized_name, [])
            if not goals:
                continue
            evaluations = self.get_constraint_evaluations_for_structure(normalized_name, goals)

            for goal_index, goal in enumerate(goals):
                evaluation = evaluations[goal_index] if goal_index < len(evaluations) else None
                note_key = self.get_constraint_note_key(normalized_name, goal)
                note_text = self.compose_constraint_note_text(
                    self.get_computed_constraint_note_text(normalized_name, goals, goal_index, evaluation),
                    self.constraint_notes.get(note_key, ""),
                )
                rows.append(
                    {
                        "structure": structure.name if goal_index == 0 else "",
                        "metric": goal.metric,
                        "goal": f"{goal.comparator.strip()} {goal.value_text.strip()}".strip(),
                        "result": evaluation.actual_text if evaluation is not None else "",
                        "notes": note_text,
                        "result_class": (
                            "result-pass"
                            if evaluation is not None and evaluation.status == "pass"
                            else "result-variation"
                            if evaluation is not None and evaluation.status == "variation"
                            else "result-fail"
                            if evaluation is not None and evaluation.status == "fail"
                            else ""
                        ),
                    }
                )
        return rows

    def build_targets_report_rows(self) -> List[Dict[str, object]]:
        if self.rtstruct is None or self.ct is None:
            return []

        rows: List[Dict[str, object]] = []
        for row in self.get_target_table_rows():
            note_key = self.get_target_note_key_for_row(row)
            rows.append(
                {
                    "ptv": str(row.get("display_name", row.get("structure_name", ""))),
                    "coverage": str(row.get("coverage_text", "")),
                    "minimum_dose": str(row.get("minimum_dose_text", "")),
                    "maximum_dose": str(row.get("maximum_dose_text", "")),
                    "notes": self.compose_target_note_text(
                        str(row.get("notes_text", "")),
                        self.target_notes.get(note_key, ""),
                    ),
                    "is_primary_ptv": bool(row.get("is_primary_ptv", False)),
                }
            )
        return rows

    def _report_html_text(self, text: object) -> str:
        return html.escape(str(text or "")).replace("\n", "<br>")

    def _build_report_table_html(
        self,
        headers: List[Tuple[str, str]],
        rows: List[Dict[str, object]],
        *,
        column_widths: Optional[List[str]] = None,
        cell_class_keys: Optional[Dict[str, str]] = None,
    ) -> str:
        if not rows:
            return '<p class="empty-note">No data available.</p>'

        parts = [
            '<table width="100%" cellspacing="0" cellpadding="0" '
            'style="width:100%; border-collapse:collapse; margin:0 0 10pt 0; table-layout:fixed;">'
        ]
        parts.append("<thead><tr>")
        for index, (_key, label) in enumerate(headers):
            width_attr = ""
            if column_widths and index < len(column_widths):
                width_attr = f' width="{html.escape(column_widths[index])}"'
            parts.append(
                f'<th{width_attr} style="text-align:left; background-color:#e6e6e6; '
                'border:1px solid #cfcfcf; padding:5pt 6pt;">'
                f"{self._report_html_text(label)}</th>"
            )
        parts.append("</tr></thead><tbody>")

        for row_index, row in enumerate(rows):
            row_background = "#ffffff" if row_index % 2 == 0 else "#f7f7f7"
            parts.append("<tr>")
            for col_index, (key, _label) in enumerate(headers):
                cell_text = self._report_html_text(row.get(key, ""))
                cell_styles = [
                    f"background-color:{row_background}",
                    "border:1px solid #d9d9d9",
                    "padding:4pt 6pt",
                    "vertical-align:top",
                ]
                if key == "ptv" and not bool(row.get("is_primary_ptv", False)):
                    cell_styles.append("padding-left:18pt")
                if key == "ptv" and bool(row.get("is_primary_ptv", False)):
                    cell_styles.append("font-weight:700")
                if cell_class_keys and key in cell_class_keys:
                    class_name = str(row.get(cell_class_keys[key], "")).strip()
                    if class_name == "result-pass":
                        cell_styles[0] = "background-color:#e5f5e8"
                        cell_styles.append("font-weight:600")
                    elif class_name == "result-variation":
                        cell_styles[0] = "background-color:#fff4cc"
                        cell_styles.append("font-weight:600")
                    elif class_name == "result-fail":
                        cell_styles[0] = "background-color:#fde7e7"
                        cell_styles.append("font-weight:600")
                width_attr = ""
                if column_widths and col_index < len(column_widths):
                    width_attr = f' width="{html.escape(column_widths[col_index])}"'
                parts.append(
                    f'<td{width_attr} style="{"; ".join(cell_styles)}">{cell_text or "&nbsp;"}</td>'
                )
            parts.append("</tr>")
        parts.append("</tbody></table>")
        return "".join(parts)

    def build_dvh_legend_html(self) -> str:
        curves = self.get_visible_dvh_curves() if self.dvh_plot_items else list(self.dvh_curves)
        if not curves:
            return ""

        parts = [
            '<table width="100%" cellspacing="0" cellpadding="0" '
            'style="width:100%; border-collapse:collapse; margin-top:8pt;">'
        ]
        column_count = 3
        row_count = int(math.ceil(len(curves) / float(column_count)))
        for row_index in range(row_count):
            parts.append("<tr>")
            for column_index in range(column_count):
                curve_index = row_index + column_index * row_count
                if curve_index >= len(curves):
                    parts.append('<td width="33%" style="padding:2pt 10pt 2pt 0;">&nbsp;</td>')
                    continue
                curve = curves[curve_index]
                r, g, b = (int(value) for value in curve.color_rgb)
                swatch = (
                    f'<span style="color: rgb({r}, {g}, {b}); font-size: 14pt; font-weight: 700;">'
                    "&#9472;&#9472;&#9472;&#9472;"
                    "</span>"
                )
                parts.append(
                    f'<td width="33%" style="padding:2pt 10pt 2pt 0; vertical-align:middle;">'
                    f'{swatch}<span style="padding-left:6pt;">{self._report_html_text(curve.name)}</span></td>'
                )
            parts.append("</tr>")
        parts.append("</table>")
        return "".join(parts)

    def export_dvh_report_image(self) -> Optional[Path]:
        if not self.dvh_curves:
            return None

        temp_file = tempfile.NamedTemporaryFile(prefix="peer_dvh_report_", suffix=".png", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        exporter = ImageExporter(self.dvh_plot.plotItem)
        try:
            params = exporter.parameters()
            if "width" in params:
                params["width"] = max(int(self.dvh_plot.width() * 2), 1400)
            if "height" in params:
                params["height"] = max(int(self.dvh_plot.height() * 2), 900)
        except Exception:
            pass

        marker_visible = self.dvh_curve_marker.isVisible()
        vline_visible = self.dvh_crosshair_vline.isVisible()
        hline_visible = self.dvh_crosshair_hline.isVisible()
        self.dvh_curve_marker.setVisible(False)
        self.dvh_crosshair_vline.setVisible(False)
        self.dvh_crosshair_hline.setVisible(False)
        try:
            image = exporter.export(toBytes=True)
            if image is None or image.isNull():
                return None
            image.save(str(temp_path))
            return temp_path
        finally:
            self.dvh_curve_marker.setVisible(marker_visible)
            self.dvh_crosshair_vline.setVisible(vline_visible)
            self.dvh_crosshair_hline.setVisible(hline_visible)

    def build_report_html(self, dvh_image_path: Optional[Path]) -> str:
        patient_lines = list(self.patient_plan_lines or ())
        patient_name = patient_lines[0] if patient_lines else "Patient name unavailable"
        patient_id = patient_lines[1] if len(patient_lines) > 1 else "ID unavailable"
        prescription = patient_lines[2] if len(patient_lines) > 2 else "Prescription unavailable"
        extra_header_lines = patient_lines[3:]
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        constraints_html = self._build_report_table_html(
            [
                ("structure", "OAR"),
                ("metric", "Metric"),
                ("goal", "Goal"),
                ("result", "Result"),
                ("notes", "Notes"),
            ],
            self.build_constraints_report_rows(),
            column_widths=["18%", "14%", "14%", "14%", "40%"],
            cell_class_keys={"result": "result_class"},
        )
        targets_html = self._build_report_table_html(
            [
                ("ptv", "PTV"),
                ("coverage", "Coverage @ Rx"),
                ("minimum_dose", "Min Dose"),
                ("maximum_dose", "Max Dose"),
                ("notes", "Notes"),
            ],
            self.build_targets_report_rows(),
            column_widths=["18%", "18%", "8%", "8%", "48%"],
        )
        extra_lines_html = "".join(
            f"<div>{self._report_html_text(line)}</div>" for line in extra_header_lines if str(line).strip()
        )
        constraint_set = self.constraints_sheet_name or NO_CONSTRAINTS_SHEET_LABEL
        dvh_legend_html = self.build_dvh_legend_html()
        if dvh_image_path is not None:
            dvh_section_html = (
                '<div class="page-break"></div><h2>DVH</h2>'
                f'<img class="dvh-image" src="{QtCore.QUrl.fromLocalFile(str(dvh_image_path)).toString()}" />'
                f"{dvh_legend_html}"
            )
        else:
            dvh_section_html = (
                '<div class="page-break"></div><h2>DVH</h2>'
                '<p class="empty-note">No DVH curves are available for this report.</p>'
            )

        return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 9pt;
  color: #111;
  margin: 0;
  padding: 0;
}}
h1 {{
  font-size: 18pt;
  margin: 0 0 8pt 0;
}}
h2 {{
  font-size: 13pt;
  margin: 14pt 0 6pt 0;
}}
.report-meta {{
  margin-bottom: 8pt;
  line-height: 1.35;
  width: 100%;
}}
.patient-name {{
  font-size: 16pt;
  font-weight: 700;
}}
.constraint-set {{
  margin-top: 4pt;
}}
.empty-note {{
  color: #555;
  font-style: italic;
}}
.page-break {{
  page-break-before: always;
}}
.dvh-image {{
  width: 100%;
  max-width: 100%;
}}
</style>
</head>
<body>
  <h1>Peer Review Report</h1>
  <div class="report-meta">
    <div class="patient-name">{self._report_html_text(patient_name)}</div>
    <div>{self._report_html_text(patient_id)}</div>
    <div>{self._report_html_text(prescription)}</div>
    {extra_lines_html}
    <div class="constraint-set"><strong>Constraint set:</strong> {self._report_html_text(constraint_set)}</div>
    <div><strong>Generated:</strong> {self._report_html_text(generated_at)}</div>
  </div>

  <h2>Constraints</h2>
  {constraints_html}

  <h2>Target Metrics</h2>
  {targets_html}

  {dvh_section_html}
</body>
</html>
"""

    def on_print_report(self):
        if self.current_patient_folder is None or self.rtstruct is None:
            QtWidgets.QMessageBox.information(self, "No patient folder", "Load a patient folder before printing a report.")
            return

        default_path = self.get_default_report_path()
        selected_path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save PDF Report",
            str(default_path),
            "PDF Files (*.pdf)",
        )
        if not selected_path:
            return

        output_path = Path(selected_path)
        if output_path.suffix.lower() != ".pdf":
            output_path = output_path.with_suffix(".pdf")

        self.show_progress_status("Generating PDF report", pump_events=True)
        dvh_image_path: Optional[Path] = None
        try:
            dvh_image_path = self.export_dvh_report_image()
            document = QtGui.QTextDocument(self)
            document.setDocumentMargin(0.0)
            document.setHtml(self.build_report_html(dvh_image_path))

            printer = QtPrintSupport.QPrinter(QtPrintSupport.QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QtPrintSupport.QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(str(output_path))
            printer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.PageSizeId.Letter))
            printer.setPageMargins(
                QtCore.QMarginsF(8.0, 8.0, 8.0, 8.0),
                QtGui.QPageLayout.Unit.Millimeter,
            )
            document.setPageSize(printer.pageRect(QtPrintSupport.QPrinter.Unit.Point).size())
            document.print_(printer)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Print failed", str(exc))
            return
        finally:
            self.clear_progress_status("Generating PDF report")
            if dvh_image_path is not None:
                try:
                    dvh_image_path.unlink(missing_ok=True)
                except OSError:
                    pass

        self.statusBar().showMessage(f"Saved PDF report to {output_path.name}", 4000)

    def set_heavy_view_updates_enabled(self, enabled: bool) -> None:
        for widget in (
            getattr(self, "axial_graphics_widget", None),
            getattr(self, "sagittal_graphics_widget", None),
            getattr(self, "coronal_graphics_widget", None),
            getattr(self, "dvh_plot", None),
        ):
            if widget is not None:
                widget.setUpdatesEnabled(enabled)

    def show_progress_status(self, message: str, *, pump_events: bool = False) -> None:
        self.progress_status_label.setText(message)
        self.statusBar().layout().activate()
        self.progress_status_label.updateGeometry()
        self.statusBar().updateGeometry()
        self.progress_status_label.repaint()
        self.statusBar().repaint()
        QtCore.QCoreApplication.sendPostedEvents(self.progress_status_label, int(QtCore.QEvent.Type.Paint))
        QtCore.QCoreApplication.sendPostedEvents(self.statusBar(), int(QtCore.QEvent.Type.Paint))
        if pump_events:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def clear_progress_status(self, expected_message: Optional[str] = None) -> None:
        if expected_message is None or self.progress_status_label.text() == expected_message:
            self.progress_status_label.clear()
            self.statusBar().layout().activate()
            self.progress_status_label.repaint()
            self.statusBar().repaint()
            QtCore.QCoreApplication.sendPostedEvents(self.progress_status_label, int(QtCore.QEvent.Type.Paint))
            QtCore.QCoreApplication.sendPostedEvents(self.statusBar(), int(QtCore.QEvent.Type.Paint))

    def write_latest_timing_report(self, error_message: Optional[str] = None) -> Optional[Path]:
        if not self.latest_timing_entries or self.latest_timing_folder is None:
            return None

        return self.write_load_timing_report(
            self.latest_timing_folder,
            self.latest_timing_entries,
            csv_path=self.latest_timing_csv_path,
            rtstruct_path=self.latest_timing_rtstruct_path,
            rtdose_paths=self.latest_timing_rtdose_paths,
            error_message=error_message,
        )

    def update_background_dvh_timing_report(self, duration_s: float, error_message: Optional[str] = None):
        if not self.latest_timing_entries or self.latest_timing_folder is None:
            return

        updated_entries: List[Tuple[str, Optional[float]]] = []
        updated = False
        for label, recorded_duration in self.latest_timing_entries:
            if label == "Compute DVH (background)":
                updated_entries.append((label, duration_s))
                updated = True
            else:
                updated_entries.append((label, recorded_duration))

        if not updated:
            return

        self.latest_timing_entries = updated_entries
        self.write_latest_timing_report(error_message=error_message)

    def reset_dvh_plot(self):
        plot_item = self.dvh_plot.getPlotItem()
        plot_item.clear()
        if plot_item.legend is not None:
            plot_item.legend.scene().removeItem(plot_item.legend)
            plot_item.legend = None
        self.dvh_plot_items = {}
        self.dvh_curve_marker.setData([], [])
        self.dvh_crosshair_vline.setVisible(False)
        self.dvh_crosshair_hline.setVisible(False)
        plot_item.addItem(self.dvh_crosshair_vline, ignoreBounds=True)
        plot_item.addItem(self.dvh_crosshair_hline, ignoreBounds=True)
        plot_item.addItem(self.dvh_curve_marker)

    def get_current_dvh_view_range(
        self,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if not self.dvh_plot_items:
            return None
        x_range, y_range = self.dvh_plot.getPlotItem().vb.viewRange()
        return (
            (float(x_range[0]), float(x_range[1])),
            (float(y_range[0]), float(y_range[1])),
        )

    def get_visible_dvh_curves(self) -> List[DVHCurve]:
        return get_visible_dvh_curves_helper(self.dvh_curves, self.dvh_structure_is_visible)

    def fit_dvh_view_to_visible_curves(self) -> None:
        view_range = get_visible_dvh_view_range(self.dvh_curves, self.dvh_structure_is_visible)
        if view_range is None:
            return
        x_range, y_range = view_range
        self.dvh_plot.setXRange(x_range[0], x_range[1], padding=0.02)
        self.dvh_plot.setYRange(y_range[0], y_range[1], padding=0.02)

    def get_autoscroll_speed_mm_per_s(self) -> Optional[float]:
        slice_step_mm = self.get_autoscroll_slice_step_mm()
        if slice_step_mm is None:
            return None
        interval_s = max(float(self.autoscroll_timer.interval()), 1.0) / 1000.0
        return float(slice_step_mm) / interval_s

    def update_autoscroll_speed_label(self) -> None:
        if not hasattr(self, "autoscroll_speed_label"):
            return
        speed_mm_per_s = self.get_autoscroll_speed_mm_per_s()
        if speed_mm_per_s is None:
            self.autoscroll_speed_label.setText("-- mm/s")
        else:
            self.autoscroll_speed_label.setText(f"{speed_mm_per_s:.1f} mm/s")
        self.autoscroll_speed_label.adjustSize()
        self.update_axial_overlay_positions()

    def get_cached_ptv_coverage_text(self, normalized_name: str) -> Optional[str]:
        cached_target_rows = self.cached_target_table_rows or []
        for row in cached_target_rows:
            if (
                bool(row.get("is_primary_ptv", False))
                and str(row.get("normalized_name", "")) == normalized_name
            ):
                coverage_text = str(row.get("coverage_text", "")).strip()
                if coverage_text:
                    return coverage_text
        return self.dvh_ptv_coverage_cache.get(normalized_name)

    def update_dvh_secondary_metric_caches(self) -> None:
        for curve in self.dvh_curves:
            normalized_name = normalize_structure_name(curve.name)
            structure = self.get_structure_by_normalized_name(normalized_name)
            cached_volume_cc = self.structure_geometry_volume_cache.get(normalized_name)
            if cached_volume_cc is not None:
                self.dvh_structure_volume_cache[normalized_name] = cached_volume_cc
            elif structure is not None and not self.restore_saved_results_without_calculation:
                self.dvh_structure_volume_cache[normalized_name] = self.get_structure_geometry_volume_cc(structure)
            else:
                self.dvh_structure_volume_cache[normalized_name] = float(curve.volume_cc)

        cached_target_rows = self.cached_target_table_rows or []
        for row in cached_target_rows:
            if not bool(row.get("is_primary_ptv", False)):
                continue
            normalized_name = str(row.get("normalized_name", ""))
            coverage_text = str(row.get("coverage_text", "")).strip()
            if normalized_name and coverage_text:
                self.dvh_ptv_coverage_cache[normalized_name] = coverage_text

    def update_dvh_goal_evaluation_cache(
        self,
        evaluations_by_name: Optional[Dict[str, List[StructureGoalEvaluation]]] = None,
    ) -> None:
        source = evaluations_by_name if evaluations_by_name is not None else self.structure_goal_evaluations
        for normalized_name, evaluations in source.items():
            if evaluations:
                self.dvh_structure_goal_evaluation_cache[normalized_name] = list(evaluations)

    def clear_dvh_curve_selection(self):
        self.selected_dvh_curve_name = None
        self.dvh_curve_marker.setData([], [])
        self.dvh_crosshair_vline.setVisible(False)
        self.dvh_crosshair_hline.setVisible(False)
        self.dvh_readout_label.setText("Click a DVH curve to inspect dose and volume.")
        self.update_dvh_curve_highlighting()
        self.update_dvh_cache_button()

    def on_clear_dvh_curve_shortcut(self):
        if self.selected_dvh_curve_name is None:
            return
        self.clear_dvh_curve_selection()

    def get_curve_for_name(self, normalized_name: str) -> Optional[DVHCurve]:
        return get_curve_for_name_helper(self.dvh_curves, normalized_name)

    def update_dvh_curve_highlighting(self):
        for normalized_name, item in self.dvh_plot_items.items():
            curve = self.get_curve_for_name(normalized_name)
            if curve is None:
                continue
            width = get_dvh_curve_highlight_width(normalized_name, self.selected_dvh_curve_name)
            item.setPen(pg.mkPen(color=curve.color_rgb, width=width))

    def on_dvh_curve_clicked(self, item: pg.PlotDataItem, event):
        normalized_name = ""
        for name, plot_item in self.dvh_plot_items.items():
            if plot_item is item:
                normalized_name = name
                break
        if not normalized_name:
            return
        self.select_dvh_curve(normalized_name, event.scenePos() if event is not None and hasattr(event, "scenePos") else None)

    def select_dvh_curve(self, normalized_name: str, scene_pos: Optional[QtCore.QPointF] = None):
        self.selected_dvh_curve_name = normalized_name
        self.update_dvh_curve_highlighting()
        if scene_pos is not None:
            self.update_dvh_curve_readout(scene_pos)
        else:
            curve = self.get_curve_for_name(normalized_name)
            if curve is None:
                return
            self.dvh_readout_label.setText(get_dvh_selection_prompt(curve.name))

    def find_nearest_dvh_curve_name(self, scene_pos: QtCore.QPointF, tolerance_px: float = 16.0) -> Optional[str]:
        return find_nearest_dvh_curve_name_helper(
            self.dvh_curves,
            self.dvh_plot.getPlotItem(),
            scene_pos,
            tolerance_px=tolerance_px,
        )

    def update_dvh_curve_readout(self, scene_pos: QtCore.QPointF):
        if self.selected_dvh_curve_name is None:
            self.dvh_curve_marker.setData([], [])
            self.dvh_crosshair_vline.setVisible(False)
            self.dvh_crosshair_hline.setVisible(False)
            return

        curve = self.get_curve_for_name(self.selected_dvh_curve_name)
        if curve is None or curve.dose_bins_gy.size == 0:
            self.clear_dvh_curve_selection()
            return

        plot_item = self.dvh_plot.getPlotItem()
        readout_state = build_dvh_readout_state(curve, plot_item, scene_pos)
        if readout_state is None:
            self.dvh_curve_marker.setData([], [])
            self.dvh_crosshair_vline.setVisible(False)
            self.dvh_crosshair_hline.setVisible(False)
            self.dvh_readout_label.setText(get_dvh_selection_prompt(curve.name))
            return

        self.dvh_curve_marker.setData(
            [readout_state.dose_gy],
            [readout_state.volume_pct],
            pen=pg.mkPen(curve.color_rgb, width=2),
            brush=pg.mkBrush(curve.color_rgb),
        )
        self.dvh_crosshair_vline.setValue(readout_state.dose_gy)
        self.dvh_crosshair_hline.setValue(readout_state.volume_pct)
        self.dvh_crosshair_vline.setVisible(True)
        self.dvh_crosshair_hline.setVisible(True)
        self.dvh_readout_label.setText(readout_state.text)

    def on_dvh_mouse_moved(self, event):
        if isinstance(event, tuple):
            if not event:
                return
            scene_pos = event[0]
        else:
            scene_pos = event
        self.update_dvh_curve_readout(scene_pos)

    def on_dvh_plot_clicked(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        normalized_name = self.find_nearest_dvh_curve_name(event.scenePos())
        if normalized_name is None:
            return
        self.select_dvh_curve(normalized_name, event.scenePos())

    def get_dvh_mode(self) -> str:
        return "high_accuracy"

    def get_dvh_mode_label(self) -> str:
        return "High accuracy"

    def render_dvh_plot(self, *, reset_view: bool = False):
        if self.tabs.currentWidget() is not self.dvh_tab:
            self.dvh_plot_refresh_pending = True
            return

        self.dvh_plot_refresh_pending = False
        previous_view_range = None if reset_view else self.get_current_dvh_view_range()
        self.reset_dvh_plot()
        if not self.dvh_curves:
            self.clear_dvh_curve_selection()
            return

        self.selected_dvh_curve_name = resolve_selected_curve_name(
            self.dvh_curves,
            self.selected_dvh_curve_name,
            self.dvh_structure_is_visible,
        )

        for spec in build_dvh_plot_curve_specs(
            self.dvh_curves,
            self.dvh_structure_is_visible,
            self.selected_dvh_curve_name,
        ):
            item = self.dvh_plot.plot(
                spec.plot_dose_bins,
                spec.plot_volume_pct,
                pen=pg.mkPen(color=spec.color_rgb, width=spec.width),
            )
            item.setCurveClickable(True, width=16)
            item.sigClicked.connect(self.on_dvh_curve_clicked)
            self.dvh_plot_items[spec.normalized_name] = item

        if self.dvh_plot_items:
            if previous_view_range is not None:
                x_range, y_range = previous_view_range
                self.dvh_plot.getPlotItem().vb.setRange(
                    xRange=x_range,
                    yRange=y_range,
                    padding=0.0,
                )
            else:
                self.fit_dvh_view_to_visible_curves()

    def get_dvh_plot_arrays(self, curve: DVHCurve) -> Tuple[np.ndarray, np.ndarray]:
        return get_dvh_plot_arrays_helper(curve)

    def reset_dvh_refresh_state(
        self,
        status_text: str,
        *,
        clear_progress_label: Optional[str] = None,
    ) -> bool:
        self.dvh_job_manager.invalidate()
        self.structure_mask_cache = None
        self.structure_mask_cache_names = []
        self.dvh_request_structure_names = {}
        self.dvh_curves = []
        self.structure_goal_evaluations = {}
        self.update_structure_list_goal_texts()
        self.update_dvh_cache_button()
        self.dvh_status_label.setText(status_text)
        self.clear_dvh_curve_selection()
        self.render_dvh_plot()
        if clear_progress_label is not None:
            self.clear_progress_status(clear_progress_label)
        return False

    def refresh_dvh(self):
        if self.ct is None or self.dose is None or self.rtstruct is None:
            return self.reset_dvh_refresh_state(get_dvh_missing_inputs_status_text())

        self.show_progress_status("Computing DVHs")
        selected_names = self.get_selected_dvh_structure_names()
        selected_rtstruct = self.build_selected_dvh_rtstruct(selected_names)
        refresh_request = build_dvh_refresh_request(
            selected_names,
            selected_rtstruct,
            self.structure_mask_cache,
            self.structure_mask_cache_names,
        )
        if refresh_request is None:
            return self.reset_dvh_refresh_state(
                get_dvh_no_selection_status_text(),
                clear_progress_label="Computing DVHs",
            )

        self.render_dvh_plot()

        request_id = self.dvh_job_manager.start(
            self.ct,
            self.dose,
            selected_rtstruct,
            self.sampled_dose_volume_ct,
            refresh_request.reusable_mask_cache,
            self.get_dvh_mode(),
        )
        self.dvh_request_structure_names = {request_id: list(refresh_request.selected_names)}
        return True

    def on_dvh_task_finished(
        self,
        request_id: int,
        curves: object,
        mask_cache: object,
        duration_s: float,
    ):
        request_structure_names = self.dvh_request_structure_names.pop(request_id, [])
        if not self.dvh_job_manager.is_current(request_id):
            return

        self.structure_mask_cache = mask_cache
        self.structure_mask_cache_names = list(request_structure_names)
        self.dvh_curves = list(curves)
        self.show_progress_status("Computing metrics")
        self.refresh_visible_structure_goal_evaluations()
        self.update_dvh_goal_evaluation_cache()
        self.update_dvh_secondary_metric_caches()
        self.cached_target_table_rows = None
        self.update_structure_list_goal_texts()
        completion_state = build_dvh_task_completion_state(
            self.dvh_curves,
            self.selected_dvh_curve_name,
            self.dvh_structure_is_visible,
        )
        self.selected_dvh_curve_name = completion_state.selected_curve_name
        if completion_state.status_text is None:
            self.dvh_status_label.clear()
        else:
            self.dvh_status_label.setText(completion_state.status_text)
        if completion_state.should_clear_selection:
            self.clear_dvh_curve_selection()
        self.render_dvh_plot()
        self.update_dvh_cache_button()
        self.update_background_dvh_timing_report(duration_s)
        self.clear_progress_status("Computing metrics")

    def on_dvh_task_failed(self, request_id: int, error_message: str, duration_s: float):
        self.dvh_request_structure_names.pop(request_id, None)
        if not self.dvh_job_manager.is_current(request_id):
            return

        self.dvh_status_label.setText(get_dvh_task_failed_status_text(error_message))
        self.render_dvh_plot()
        self.update_dvh_cache_button()
        self.update_background_dvh_timing_report(duration_s, error_message=error_message)
        self.clear_progress_status()

    def current_dvh_curve_names(self) -> List[str]:
        return get_current_dvh_curve_names_helper(self.dvh_curves)

    def refresh_visible_structure_goal_evaluations(
        self,
        precomputed: Optional[Dict[str, List[StructureGoalEvaluation]]] = None,
    ) -> None:
        selected_names = self.get_selected_dvh_structure_names()
        if self.restore_saved_results_without_calculation:
            selected_name_set = set(selected_names)
            source = precomputed or {}
            self.structure_goal_evaluations = {
                normalize_structure_name(name): list(evaluations)
                for name, evaluations in source.items()
                if normalize_structure_name(name) in selected_name_set
            }
            return
        self.structure_goal_evaluations = compute_visible_structure_goal_evaluations(
            self.dvh_curves,
            self.structure_goals_by_name,
            selected_names,
            precomputed=precomputed,
        )

    def on_dvh_structure_visibility_changed(self, *_args):
        selected_names = self.get_selected_dvh_structure_names()
        refresh_plan = build_dvh_visibility_refresh_plan(
            selected_names,
            self.current_dvh_curve_names(),
            self.dvh_curves,
        )
        if refresh_plan.should_refresh_from_scratch:
            self.refresh_dvh()
            return

        if refresh_plan.should_invalidate_jobs:
            self.dvh_job_manager.invalidate()
            self.dvh_request_structure_names = {}
            self.refresh_visible_structure_goal_evaluations()
            self.update_dvh_goal_evaluation_cache()
            self.update_structure_list_goal_texts()
            self.render_dvh_plot()
            if refresh_plan.should_clear_status:
                self.dvh_status_label.clear()
            return

        self.refresh_dvh()

    def on_clear_dvh_structures_clicked(self):
        changed = self.dvh_structure_list.set_checked_names([], emit_signal=False)
        if not changed:
            return
        self.on_dvh_structure_visibility_changed()

    def get_dose_display_range(self) -> Tuple[float, float]:
        if self.dose is None:
            return 0.0, 1.0

        global_max = float(np.nanmax(self.dose.dose_gy))
        lower_value, upper_value = self.dose_range_slider.values()
        min_dose = global_max * (lower_value / 1000.0)
        max_dose = global_max * (upper_value / 1000.0)
        if max_dose <= min_dose:
            max_dose = min_dose + max(global_max * 0.001, 1e-6)
        return min_dose, max_dose

    def dose_gy_to_slider_value(self, dose_gy: float) -> int:
        if self.dose is None:
            return 0
        global_max = max(float(np.nanmax(self.dose.dose_gy)), 1e-6)
        return int(round(np.clip(dose_gy / global_max, 0.0, 1.0) * 1000.0))

    def get_default_colorwash_min_dose_gy(self) -> float:
        available_phase_rx = [
            float(phase.prescription_dose_gy)
            for phase in self.plan_phases
            if phase.prescription_dose_gy > 0.0 and phase.dose_path
        ]
        ptv_dose_values = get_ptv_dose_levels_gy(self.rtstruct)

        if len(available_phase_rx) == 1:
            return available_phase_rx[0] * 0.95
        if len(available_phase_rx) > 1:
            if ptv_dose_values:
                return min(ptv_dose_values) * 0.95
            return min(available_phase_rx) * 0.95
        if ptv_dose_values:
            if len(ptv_dose_values) == 1:
                return ptv_dose_values[0] * 0.95
            return min(ptv_dose_values) * 0.95
        return 0.0

    def get_total_rx_dose_gy(self) -> Optional[float]:
        available_phase_rx = [
            float(phase.prescription_dose_gy)
            for phase in self.plan_phases
            if phase.prescription_dose_gy > 0.0 and phase.dose_path
        ]
        if available_phase_rx:
            return float(sum(available_phase_rx))

        highest_reference_gy: Optional[float] = None
        for structure in self.get_sorted_ptv_structures():
            normalized_name = normalize_structure_name(structure.name)
            reference_gy = self.get_stereotactic_threshold_gy(normalized_name, structure.name)
            if reference_gy is None or reference_gy <= 0.0:
                continue
            if highest_reference_gy is None or reference_gy > highest_reference_gy:
                highest_reference_gy = reference_gy
        return highest_reference_gy

    def apply_default_dose_range(self):
        min_dose_gy = self.get_default_colorwash_min_dose_gy()
        blocker = QtCore.QSignalBlocker(self.dose_range_slider)
        self.dose_range_slider.setValues(self.dose_gy_to_slider_value(min_dose_gy), 1000)
        del blocker

    def update_dose_range_controls(self):
        interaction_enabled = not self.autoscroll_button.isChecked()
        enabled = self.dose is not None and interaction_enabled
        max_enabled = self.sampled_dose_volume_ct is not None and interaction_enabled
        self.dose_opacity_slider.setEnabled(enabled)
        self.dose_toggle_button.setEnabled(enabled)
        self.dose_min_edit.setEnabled(enabled)
        self.dose_max_edit.setEnabled(enabled)
        self.dose_range_slider.setEnabled(enabled)
        self.max_dose_button.setEnabled(max_enabled)

        if self.dose is None:
            for widget, text in ((self.dose_min_edit, "0.00"), (self.dose_max_edit, "0.00")):
                blocker = QtCore.QSignalBlocker(widget)
                widget.setText(text)
                del blocker
            self.dose_range_label.setText("Dose range: 0.00 Gy - 100%")
            return

        global_max = float(np.nanmax(self.dose.dose_gy))
        min_dose, max_dose = self.get_dose_display_range()
        max_pct = 100.0 * max_dose / max(global_max, 1e-6)

        for widget, value in ((self.dose_min_edit, min_dose), (self.dose_max_edit, max_dose)):
            blocker = QtCore.QSignalBlocker(widget)
            widget.setText(f"{value:.2f}")
            del blocker

        self.dose_range_label.setText(f"Dose range: {min_dose:.2f} Gy - {max_dose:.2f} Gy ({max_pct:.1f}%)")

    def current_dose_alpha(self) -> float:
        if not self.dose_overlay_enabled:
            return 0.0
        return self.dose_opacity_slider.value() / 100.0

    def on_dose_range_slider_changed(self, lower_value: int, upper_value: int):
        if upper_value < lower_value:
            self.dose_range_slider.setValues(lower_value, lower_value)
            return
        self.update_dose_range_controls()
        self.refresh_all_views()

    def on_dose_editing_finished(self):
        if self.dose is None:
            self.update_dose_range_controls()
            return

        try:
            requested_min = float(self.dose_min_edit.text() or "0")
            requested_max = float(self.dose_max_edit.text() or "0")
        except ValueError:
            self.update_dose_range_controls()
            return

        min_value = self.dose_gy_to_slider_value(requested_min)
        max_value = self.dose_gy_to_slider_value(requested_max)
        if max_value < min_value:
            sender = self.sender()
            if sender is self.dose_min_edit:
                max_value = min_value
            else:
                min_value = max_value

        self.dose_range_slider.setValues(min_value, max_value)
        self.update_dose_range_controls()
        self.refresh_all_views()

    def on_toggle_autoscroll(self, checked: bool):
        if checked:
            self.statusBar().clearMessage()
            visible_range = self.get_visible_structure_slice_range()
            if visible_range is None:
                QtWidgets.QMessageBox.information(
                    self,
                    "No visible PTVs",
                    "Select at least one visible PTV to auto scroll through its superior-inferior extent.",
                )
                self.cancel_autoscroll()
                return

            start_idx, _ = visible_range
            self.autoscroll_direction = 1
            self.set_autoscroll_ui_locked(True)
            self.slice_slider.setValue(start_idx)
            self.autoscroll_timer.start()
        else:
            self.statusBar().clearMessage()
            self.reset_autoscroll_ui()

    def advance_autoscroll(self):
        visible_range = self.get_visible_structure_slice_range()
        if visible_range is None:
            self.cancel_autoscroll()
            return

        start_idx, end_idx = visible_range
        current = self.slice_slider.value()

        if current >= end_idx:
            self.autoscroll_direction = -1
        elif current <= start_idx:
            self.autoscroll_direction = 1

        next_value = current + self.autoscroll_direction
        next_value = max(start_idx, min(end_idx, next_value))
        self.slice_slider.setValue(next_value)
        if self.autoscroll_button.isChecked():
            self.autoscroll_timer.start()

    def get_visible_structure_slice_range(self) -> Optional[Tuple[int, int]]:
        if self.rtstruct is None or self.ct is None:
            return None

        indices: List[int] = []
        for idx, structure in enumerate(self.rtstruct.structures):
            if not self.structure_is_visible(idx):
                continue
            if not normalize_structure_name(structure.name).startswith("PTV"):
                continue
            indices.extend(structure.points_rc_by_slice.keys())

        if not indices:
            return None

        structure_start = min(indices)
        structure_end = max(indices)
        margin_mm = 10.0
        z_positions = self.ct.z_positions_mm

        start_pos = float(z_positions[structure_start] - margin_mm)
        end_pos = float(z_positions[structure_end] + margin_mm)

        start_idx = int(np.searchsorted(z_positions, start_pos, side="left"))
        end_idx = int(np.searchsorted(z_positions, end_pos, side="right") - 1)
        start_idx = max(0, min(start_idx, len(z_positions) - 1))
        end_idx = max(0, min(end_idx, len(z_positions) - 1))

        return start_idx, end_idx

    def on_structure_visibility_changed(self, *_args):
        self.refresh_all_views()

        if not self.autoscroll_button.isChecked():
            return

        visible_range = self.get_visible_structure_slice_range()
        if visible_range is None:
            self.cancel_autoscroll()
            return

        start_idx, end_idx = visible_range
        current = int(np.clip(self.slice_slider.value(), start_idx, end_idx))
        if current != self.slice_slider.value():
            self.slice_slider.setValue(current)

        if current >= end_idx:
            self.autoscroll_direction = -1
        elif current <= start_idx:
            self.autoscroll_direction = 1

    def format_structure_goal_line(self, evaluation: StructureGoalEvaluation) -> str:
        goal_clause = " ".join(part for part in (evaluation.metric, evaluation.comparator, evaluation.goal_text) if part).strip()
        return f"{goal_clause}: {evaluation.actual_text}"

    def structure_goal_line_color(self, evaluation: StructureGoalEvaluation) -> Optional[str]:
        if evaluation.status == "pass" or evaluation.passed is True:
            return "#63c174"
        if evaluation.status == "variation":
            return "#ffd54a"
        if evaluation.status == "fail" or evaluation.passed is False:
            return "#ff6b6b"
        return None

    def edit_constraint_note(self, note_key: str, title: str):
        existing_note = self.constraint_notes.get(note_key, "")
        note_text, accepted = QtWidgets.QInputDialog.getMultiLineText(
            self,
            "Constraint Note",
            title,
            existing_note,
        )
        if not accepted:
            return
        cleaned_note = note_text.strip()
        if cleaned_note:
            self.constraint_notes[note_key] = cleaned_note
        else:
            self.constraint_notes.pop(note_key, None)
        self.update_constraints_table()
        self.statusBar().showMessage("Constraint note updated. Press Save to persist it.", 4000)

    def create_target_note_button(
        self,
        note_key: str,
        title: str,
        background_color: Optional[QtGui.QColor] = None,
    ) -> QtWidgets.QWidget:
        return create_target_note_button_widget_helper(
            note_key,
            title,
            self.target_notes.get(note_key, ""),
            background_color,
            on_edit_note=self.edit_target_note,
        )

    def edit_target_note(self, note_key: str, title: str):
        existing_note = self.target_notes.get(note_key, "")
        note_text, accepted = QtWidgets.QInputDialog.getMultiLineText(
            self,
            "Target Note",
            title,
            existing_note,
        )
        if not accepted:
            return
        cleaned_note = note_text.strip()
        if cleaned_note:
            self.target_notes[note_key] = cleaned_note
        else:
            self.target_notes.pop(note_key, None)
        self.update_targets_table()
        self.statusBar().showMessage("Target note updated. Press Save to persist it.", 4000)

    def update_constraints_table_column_widths(self):
        if self.constraints_table.columnCount() != 6:
            return

        column_widths = get_constraints_table_column_widths(self.constraints_table.viewport().width())

        for column_index, width in enumerate(column_widths):
            self.constraints_table.setColumnWidth(column_index, max(1, width))

    def parse_ptv_rx_gy_from_name(self, structure_name: str) -> Optional[float]:
        normalized_name = normalize_structure_name(structure_name)
        if not normalized_name.startswith("PTV"):
            return None

        digits = "".join(ch for ch in normalized_name if ch.isdigit())
        if not digits:
            return None
        return float(int(digits)) / 100.0

    def get_sorted_ptv_structures(self) -> List[StructureSliceContours]:
        return get_sorted_ptv_structures_helper(
            self.rtstruct,
            is_listable_structure_name=self.is_listable_structure_name,
            parse_ptv_rx_gy_from_name=self.parse_ptv_rx_gy_from_name,
        )

    def is_nested_target_structure_name(self, normalized_name: str) -> bool:
        return normalized_name.startswith(("PTV", "GTV", "CTV"))

    def get_target_structure_slice_masks(self, structure: StructureSliceContours) -> Dict[int, np.ndarray]:
        if self.ct is None:
            return {}

        normalized_name = normalize_structure_name(structure.name)
        cached_masks = self.target_slice_mask_cache.get(normalized_name)
        if cached_masks is not None:
            return cached_masks

        cached_volume_mask = self.structure_volume_mask_cache.get(normalized_name)
        if cached_volume_mask is not None:
            structure_masks = {
                int(slice_index): np.asarray(cached_volume_mask[slice_index], dtype=bool)
                for slice_index in np.nonzero(np.any(cached_volume_mask, axis=(1, 2)))[0]
            }
            self.target_slice_mask_cache[normalized_name] = structure_masks
            return structure_masks

        structure_masks: Dict[int, np.ndarray] = {}
        for slice_index in sorted(structure.points_rc_by_slice):
            mask = build_structure_slice_mask(structure, slice_index, self.ct.rows, self.ct.cols)
            if np.any(mask):
                structure_masks[slice_index] = mask
        self.target_slice_mask_cache[normalized_name] = structure_masks
        return structure_masks

    def get_ptv_union_slice_masks(self) -> Dict[int, np.ndarray]:
        if self.ptv_union_slice_mask_cache is not None:
            return self.ptv_union_slice_mask_cache
        if self.ptv_union_volume_mask_cache is not None:
            self.ptv_union_slice_mask_cache = {
                int(slice_index): np.asarray(self.ptv_union_volume_mask_cache[slice_index], dtype=bool)
                for slice_index in np.nonzero(np.any(self.ptv_union_volume_mask_cache, axis=(1, 2)))[0]
            }
            return self.ptv_union_slice_mask_cache
        if self.rtstruct is None or self.ct is None:
            self.ptv_union_slice_mask_cache = {}
            return self.ptv_union_slice_mask_cache

        union_masks: Dict[int, np.ndarray] = {}
        for structure in self.rtstruct.structures:
            if not normalize_structure_name(structure.name).startswith("PTV"):
                continue
            for slice_index, mask in self.get_target_structure_slice_masks(structure).items():
                existing_mask = union_masks.get(slice_index)
                if existing_mask is None:
                    union_masks[slice_index] = np.asarray(mask, dtype=bool).copy()
                else:
                    existing_mask |= mask

        for slice_index, mask in list(union_masks.items()):
            if not np.any(mask):
                union_masks.pop(slice_index, None)
                continue
            normalized_mask = fill_binary_holes_2d(mask)
            if binary_dilation is not None:
                normalized_mask = np.asarray(
                    binary_dilation(normalized_mask, structure=np.ones((3, 3), dtype=bool)),
                    dtype=bool,
                )
            union_masks[slice_index] = np.asarray(normalized_mask, dtype=bool)

        self.ptv_union_slice_mask_cache = union_masks
        return union_masks

    def get_ptv_union_volume_mask(self) -> Optional[np.ndarray]:
        if self.ptv_union_volume_mask_cache is not None:
            return self.ptv_union_volume_mask_cache
        if self.ct is None:
            return None

        volume_mask = np.zeros(self.ct.volume_hu.shape, dtype=bool)
        for slice_index, slice_mask in self.get_ptv_union_slice_masks().items():
            if 0 <= slice_index < volume_mask.shape[0]:
                volume_mask[slice_index] = slice_mask
        self.ptv_union_volume_mask_cache = volume_mask
        return volume_mask

    def get_local_structure_mask(
        self,
        structure: StructureSliceContours,
        z_start: int,
        z_end: int,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> np.ndarray:
        local_shape = (
            z_end - z_start + 1,
            row_end - row_start + 1,
            col_end - col_start + 1,
        )
        local_mask = np.zeros(local_shape, dtype=bool)
        for slice_index, slice_mask in self.get_target_structure_slice_masks(structure).items():
            if slice_index < z_start or slice_index > z_end:
                continue
            local_z = slice_index - z_start
            local_mask[local_z] = slice_mask[row_start: row_end + 1, col_start: col_end + 1]
        return local_mask

    def get_structure_volume_mask(self, structure: StructureSliceContours) -> np.ndarray:
        if self.ct is None:
            return np.zeros((0, 0, 0), dtype=bool)

        normalized_name = normalize_structure_name(structure.name)
        cached_mask = self.structure_volume_mask_cache.get(normalized_name)
        if cached_mask is not None:
            return cached_mask

        volume_mask = np.zeros(self.ct.volume_hu.shape, dtype=bool)
        for slice_index, slice_mask in self.get_target_structure_slice_masks(structure).items():
            if 0 <= slice_index < volume_mask.shape[0]:
                volume_mask[slice_index] = slice_mask
        self.structure_volume_mask_cache[normalized_name] = volume_mask
        return volume_mask

    def get_structure_geometry_volume_cc(self, structure: StructureSliceContours) -> float:
        if self.ct is None or self.dose is None:
            return 0.0

        normalized_name = normalize_structure_name(structure.name)
        cached_volume_cc = self.structure_geometry_volume_cache.get(normalized_name)
        if cached_volume_cc is not None:
            return cached_volume_cc

        metrics = estimate_structure_geometry_metrics(
            self.ct,
            self.dose,
            structure,
            structure_mask_cache=self.get_target_structure_slice_masks(structure),
        )
        volume_cc = float(metrics.volume_mm3 / 1000.0)
        self.structure_geometry_volume_cache[normalized_name] = volume_cc
        return volume_cc

    def get_brain_structure(self) -> Optional[StructureSliceContours]:
        if self.rtstruct is None:
            return None

        preferred_match: Optional[StructureSliceContours] = None
        fallback_match: Optional[StructureSliceContours] = None
        for structure in self.rtstruct.structures:
            normalized_name = normalize_structure_name(structure.name)
            if normalized_name == "BRAIN":
                preferred_match = structure
                break
            if normalized_name.startswith("BRAIN") and "BRAINSTEM" not in normalized_name:
                fallback_match = structure
        return preferred_match or fallback_match

    def get_stereotactic_competing_ptv_entries(
        self,
        structure: StructureSliceContours,
    ) -> List[Tuple[str, Dict[int, np.ndarray]]]:
        return get_stereotactic_competing_ptv_entries_helper(
            structure,
            sorted_ptv_structures=self.get_sorted_ptv_structures(),
            get_target_structure_slice_masks=self.get_target_structure_slice_masks,
            structure_is_fully_encompassed=self.structure_is_fully_encompassed,
        )

    def structure_is_fully_encompassed(
        self,
        parent_structure: StructureSliceContours,
        candidate_structure: StructureSliceContours,
    ) -> bool:
        if self.ct is None:
            return False

        parent_masks = self.get_target_structure_slice_masks(parent_structure)
        candidate_masks = self.get_target_structure_slice_masks(candidate_structure)
        if not parent_masks or not candidate_masks:
            return False

        for slice_index, candidate_mask in candidate_masks.items():
            parent_mask = parent_masks.get(slice_index)
            if parent_mask is None:
                return False
            if not np.all(parent_mask[candidate_mask]):
                return False
        return True

    def get_structure_mask_voxel_count(self, structure: StructureSliceContours) -> int:
        return int(
            sum(
                int(np.count_nonzero(mask))
                for mask in self.get_target_structure_slice_masks(structure).values()
            )
        )

    def get_preferred_manual_target_parent_name(
        self,
        structure: StructureSliceContours,
    ) -> Optional[str]:
        return get_preferred_manual_target_parent_name_helper(
            structure,
            additional_target_subvolume_names=self.additional_target_subvolume_names,
            sorted_ptv_structures=self.get_sorted_ptv_structures(),
            structure_is_fully_encompassed=self.structure_is_fully_encompassed,
            get_structure_mask_voxel_count=self.get_structure_mask_voxel_count,
        )

    def get_nested_target_structures(self, parent_structure: StructureSliceContours) -> List[StructureSliceContours]:
        if self.rtstruct is None:
            return []

        parent_normalized_name = normalize_structure_name(parent_structure.name)
        cached_names = self.target_containment_cache.get(parent_normalized_name)
        structures_by_name = {
            normalize_structure_name(structure.name): structure
            for structure in self.rtstruct.structures
        }
        nested_names = resolve_nested_target_names(
            parent_structure,
            rtstruct=self.rtstruct,
            cached_names=cached_names,
            additional_target_subvolume_names=self.additional_target_subvolume_names,
            is_listable_structure_name=self.is_listable_structure_name,
            is_nested_target_structure_name=self.is_nested_target_structure_name,
            parse_ptv_rx_gy_from_name=self.parse_ptv_rx_gy_from_name,
            get_preferred_manual_target_parent_name=self.get_preferred_manual_target_parent_name,
            structure_is_fully_encompassed=self.structure_is_fully_encompassed,
        )
        self.target_containment_cache[parent_normalized_name] = list(nested_names)
        return [
            structures_by_name[name]
            for name in nested_names
            if name in structures_by_name
        ]

    def get_phase_dose_volume(self, dose_path: str) -> Optional[DoseVolume]:
        if not dose_path:
            return None
        dose_volume = self.phase_dose_volumes_by_path.get(dose_path)
        if dose_volume is not None:
            return dose_volume
        dose_volume = load_rtdose(dose_path)
        self.phase_dose_volumes_by_path[dose_path] = dose_volume
        return dose_volume

    def get_phase_dose_plane(self, dose_path: str, slice_index: int) -> Optional[np.ndarray]:
        if self.ct is None:
            return None
        cache_key = (dose_path, slice_index)
        if cache_key in self.phase_dose_plane_cache:
            return self.phase_dose_plane_cache[cache_key]

        dose_volume = self.get_phase_dose_volume(dose_path)
        if dose_volume is None:
            return None

        dose_plane = sample_dose_to_ct_slice(self.ct, dose_volume, slice_index)
        self.phase_dose_plane_cache[cache_key] = dose_plane
        return dose_plane

    def get_target_dose_volume(self, source_key: str) -> Optional[DoseVolume]:
        if source_key == "combined":
            return self.dose
        return self.get_phase_dose_volume(source_key)

    def get_target_dose_volume_ct(self, source_key: str) -> Optional[np.ndarray]:
        if self.ct is None:
            return None
        if source_key == "combined":
            return self.sampled_dose_volume_ct

        cached = self.phase_dose_volume_ct_cache.get(source_key)
        if cached is not None:
            return cached

        dose_planes: List[np.ndarray] = []
        for slice_index in range(self.ct.volume_hu.shape[0]):
            dose_plane = self.get_phase_dose_plane(source_key, slice_index)
            if dose_plane is None:
                return None
            dose_planes.append(np.asarray(dose_plane, dtype=np.float32))
        if not dose_planes:
            return None

        volume_ct = np.stack(dose_planes, axis=0)
        self.phase_dose_volume_ct_cache[source_key] = volume_ct
        return volume_ct

    def get_stereotactic_volume_context(
        self,
        structure: StructureSliceContours,
        source_key: str,
        minimum_threshold_gy: float,
    ) -> Optional[Dict[str, object]]:
        if self.ct is None or self.rtstruct is None or minimum_threshold_gy <= 0.0:
            return None

        normalized_name = normalize_structure_name(structure.name)
        cache_key = (normalized_name, source_key, round(float(minimum_threshold_gy), 3))
        if cache_key in self.stereotactic_volume_context_cache:
            return self.stereotactic_volume_context_cache[cache_key]

        try:
            from scipy.ndimage import distance_transform_edt  # type: ignore
        except Exception:  # pragma: no cover - optional runtime dependency
            self.stereotactic_volume_context_cache[cache_key] = None
            return None

        dose_volume_ct = self.get_target_dose_volume_ct(source_key)
        if dose_volume_ct is None:
            self.stereotactic_volume_context_cache[cache_key] = None
            return None

        support_mask = np.asarray(dose_volume_ct >= float(minimum_threshold_gy), dtype=bool)
        if not np.any(support_mask):
            self.stereotactic_volume_context_cache[cache_key] = None
            return None

        relevant_ptv_entries = self.get_stereotactic_competing_ptv_entries(structure)
        if not relevant_ptv_entries:
            self.stereotactic_volume_context_cache[cache_key] = None
            return None

        coords = np.argwhere(support_mask)
        z_start = int(np.min(coords[:, 0]))
        z_end = int(np.max(coords[:, 0]))
        row_start = int(np.min(coords[:, 1]))
        row_end = int(np.max(coords[:, 1]))
        col_start = int(np.min(coords[:, 2]))
        col_end = int(np.max(coords[:, 2]))

        for _ptv_name, ptv_masks in relevant_ptv_entries:
            slice_indices = sorted(ptv_masks)
            if slice_indices:
                z_start = min(z_start, slice_indices[0])
                z_end = max(z_end, slice_indices[-1])
            for slice_mask in ptv_masks.values():
                coords_2d = np.argwhere(slice_mask)
                if coords_2d.size == 0:
                    continue
                row_start = min(row_start, int(np.min(coords_2d[:, 0])))
                row_end = max(row_end, int(np.max(coords_2d[:, 0])))
                col_start = min(col_start, int(np.min(coords_2d[:, 1])))
                col_end = max(col_end, int(np.max(coords_2d[:, 1])))

        dose_block = np.asarray(
            dose_volume_ct[
                z_start: z_end + 1,
                row_start: row_end + 1,
                col_start: col_end + 1,
            ],
            dtype=np.float32,
        )
        if dose_block.size == 0:
            self.stereotactic_volume_context_cache[cache_key] = None
            return None

        local_shape = dose_block.shape
        local_ptv_masks: List[np.ndarray] = []
        target_index: Optional[int] = None
        for index, (ptv_normalized_name, ptv_masks) in enumerate(relevant_ptv_entries):
            local_mask = np.zeros(local_shape, dtype=bool)
            for slice_index, slice_mask in ptv_masks.items():
                if slice_index < z_start or slice_index > z_end:
                    continue
                local_z = slice_index - z_start
                local_mask[local_z] = slice_mask[row_start: row_end + 1, col_start: col_end + 1]
            local_ptv_masks.append(local_mask)
            if ptv_normalized_name == normalized_name:
                target_index = index

        if target_index is None:
            self.stereotactic_volume_context_cache[cache_key] = None
            return None

        voxel_volume_cc = float(np.prod(self.ct.spacing_xyz_mm) / 1000.0)
        if len(local_ptv_masks) == 1:
            target_weight = np.ones(local_shape, dtype=np.float32)
        else:
            sampling = (
                float(self.ct.spacing_xyz_mm[2]),
                float(self.ct.spacing_xyz_mm[1]),
                float(self.ct.spacing_xyz_mm[0]),
            )
            eligible_distance_stack = np.stack(
                [
                    np.where(
                        local_mask,
                        distance_transform_edt(local_mask, sampling=sampling),
                        distance_transform_edt(~local_mask, sampling=sampling),
                    )
                    for local_mask in local_ptv_masks
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            min_distance = np.min(eligible_distance_stack, axis=0)
            finite_mask = np.isfinite(min_distance)
            if not np.any(finite_mask):
                self.stereotactic_volume_context_cache[cache_key] = None
                return None

            tie_mask = np.isclose(
                eligible_distance_stack,
                min_distance[None, ...],
                atol=1e-6,
            ) & np.isfinite(eligible_distance_stack)
            tie_count = np.sum(tie_mask, axis=0)
            target_weight = np.where(
                finite_mask & tie_mask[target_index],
                1.0 / np.maximum(tie_count, 1),
                0.0,
            ).astype(np.float32, copy=False)

        context: Dict[str, object] = {
            "target_weight": target_weight,
            "dose_block": dose_block,
            "voxel_volume_cc": voxel_volume_cc,
            "z_start": z_start,
            "z_end": z_end,
            "row_start": row_start,
            "row_end": row_end,
            "col_start": col_start,
            "col_end": col_end,
        }
        self.stereotactic_volume_context_cache[cache_key] = context
        return context

    def localize_stereotactic_extra_mask(
        self,
        context: Dict[str, object],
        extra_mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        return localize_stereotactic_extra_mask_helper(context, extra_mask)

    def compute_stereotactic_owned_volume_cc(
        self,
        context: Dict[str, object],
        threshold_gy: float,
        extra_mask: Optional[np.ndarray] = None,
    ) -> float:
        return compute_stereotactic_owned_volume_cc_helper(
            context,
            threshold_gy,
            extra_mask=extra_mask,
        )

    def get_target_high_accuracy_curve(
        self,
        structure: StructureSliceContours,
        source_key: str,
    ) -> Optional[DVHCurve]:
        if self.ct is None:
            return None

        normalized_name = normalize_structure_name(structure.name)
        cache_key = (normalized_name, source_key)
        if cache_key in self.target_curve_cache:
            return self.target_curve_cache[cache_key]

        dose_volume = self.get_target_dose_volume(source_key)
        if dose_volume is None:
            self.target_curve_cache[cache_key] = None
            return None

        curve = compute_single_structure_high_accuracy_curve(
            self.ct,
            dose_volume,
            structure,
            structure_mask_cache=self.get_target_structure_slice_masks(structure),
        )
        self.target_curve_cache[cache_key] = curve
        return curve

    def compute_structure_target_metric_values(
        self,
        structure: StructureSliceContours,
        threshold_gy: float,
        source_key: str,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if self.ct is None:
            return None, None, None

        normalized_name = normalize_structure_name(structure.name)
        cache_key = (normalized_name, source_key, round(float(threshold_gy), 3))
        cached = self.target_metrics_cache.get(cache_key)
        if cached is not None:
            return cached

        curve = self.get_target_high_accuracy_curve(structure, source_key)
        if curve is None or curve.volume_cc <= 0.0:
            return None, None, None

        minimum_dose_gy = float(curve.min_dose_gy)
        max_dose_gy = float(curve.max_dose_gy)
        coverage_pct = float(volume_pct_at_dose_gy(curve, threshold_gy))
        cached_values = (minimum_dose_gy, max_dose_gy, coverage_pct)
        self.target_metrics_cache[cache_key] = cached_values
        return cached_values

    def compute_structure_target_metrics(
        self,
        structure: StructureSliceContours,
        threshold_gy: float,
        source_key: str,
    ) -> Tuple[str, str, str]:
        minimum_dose_gy, max_dose_gy, coverage_pct = self.compute_structure_target_metric_values(
            structure,
            threshold_gy,
            source_key=source_key,
        )
        if minimum_dose_gy is None or max_dose_gy is None or coverage_pct is None:
            return "", "", f"@ {threshold_gy:.2f} Gy"
        return (
            self.format_target_dose_text(minimum_dose_gy, threshold_gy),
            self.format_target_dose_text(max_dose_gy, threshold_gy),
            f"{coverage_pct:.1f}% @ {threshold_gy:.2f} Gy",
        )

    def compute_dose_at_hottest_volume_cc(
        self,
        dose_values_gy: np.ndarray,
        voxel_volume_cc: float,
        target_volume_cc: float,
    ) -> float:
        if dose_values_gy.size == 0:
            return 0.0
        if voxel_volume_cc <= 0.0:
            return float(np.max(dose_values_gy))

        sorted_desc = np.sort(dose_values_gy.astype(np.float64, copy=False))[::-1]
        cumulative_volume_cc = np.arange(1, sorted_desc.size + 1, dtype=np.float64) * voxel_volume_cc
        target_cc = float(np.clip(target_volume_cc, 0.0, float(cumulative_volume_cc[-1])))
        return float(np.interp(target_cc, cumulative_volume_cc, sorted_desc))

    def format_target_dose_text(self, dose_gy: float, rx_gy: float) -> str:
        if rx_gy > 0.0:
            relative_pct = 100.0 * dose_gy / rx_gy
            return f"{relative_pct:.1f}%"
        return ""

    def get_default_stereotactic_dose_text(self, structure_name: str) -> str:
        return get_default_stereotactic_dose_text_helper(
            structure_name,
            plan_phases=self.plan_phases,
            constraints_sheet_name=self.constraints_sheet_name or "",
            phase_assignments=self.get_phase_target_assignments(),
            infer_srs_target_rx_gy=self.infer_srs_target_rx_gy_from_minimum_dose,
        )

    def infer_srs_target_rx_gy_from_minimum_dose(self, normalized_name: str) -> Optional[float]:
        if self.rtstruct is None:
            return None

        structure = next(
            (
                candidate
                for candidate in self.rtstruct.structures
                if normalize_structure_name(candidate.name) == normalized_name
            ),
            None,
        )
        if structure is None:
            return None

        source_key = "combined"
        phase_assignment = self.get_phase_target_assignments().get(normalized_name)
        if phase_assignment is not None:
            phase, _phase_rx_gy = phase_assignment
            source_key = phase.dose_path
        else:
            available_phases = [
                phase
                for phase in self.plan_phases
                if phase.prescription_dose_gy > 0.0 and phase.dose_path
            ]
            if len(available_phases) == 1:
                source_key = available_phases[0].dose_path

        curve = self.get_target_high_accuracy_curve(structure, source_key)
        if curve is None or not np.isfinite(curve.min_dose_gy):
            return None

        minimum_dose_gy = float(curve.min_dose_gy)
        candidate_rx_gy = [18.0, 21.0, 24.0]
        return min(candidate_rx_gy, key=lambda value: abs(minimum_dose_gy - value))

    def stereotactic_summary_enabled(self) -> bool:
        return stereotactic_summary_enabled_helper(self.constraints_sheet_name or "")

    def get_stereotactic_dose_text(self, normalized_name: str, structure_name: str) -> str:
        if normalized_name in self.stereotactic_target_dose_text_by_name:
            stored_text = self.stereotactic_target_dose_text_by_name[normalized_name].strip()
            if stored_text:
                return stored_text
        return self.get_default_stereotactic_dose_text(structure_name)

    def normalize_stereotactic_dose_text(self, dose_text: str) -> str:
        cleaned_text = dose_text.strip()
        if not cleaned_text:
            return ""
        try:
            dose_value = float(cleaned_text)
        except (TypeError, ValueError):
            return ""
        if dose_value <= 0.0:
            return ""
        return f"{dose_value:.2f}"

    def get_stereotactic_threshold_gy(self, normalized_name: str, structure_name: str) -> Optional[float]:
        dose_text = self.get_stereotactic_dose_text(normalized_name, structure_name)
        if not dose_text:
            return None
        try:
            threshold_gy = float(dose_text)
        except (TypeError, ValueError):
            return None
        return threshold_gy if threshold_gy > 0.0 else None

    def get_local_target_dose_block(
        self,
        source_key: str,
        z_start: int,
        z_end: int,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> Optional[np.ndarray]:
        if self.ct is None:
            return None

        if source_key == "combined":
            if self.sampled_dose_volume_ct is None:
                return None
            return np.asarray(
                self.sampled_dose_volume_ct[
                    z_start: z_end + 1,
                    row_start: row_end + 1,
                    col_start: col_end + 1,
                ],
                dtype=np.float32,
            )

        dose_planes: List[np.ndarray] = []
        for slice_index in range(z_start, z_end + 1):
            dose_plane = self.get_phase_dose_plane(source_key, slice_index)
            if dose_plane is None:
                return None
            dose_planes.append(
                np.asarray(
                    dose_plane[row_start: row_end + 1, col_start: col_end + 1],
                    dtype=np.float32,
                )
            )
        if not dose_planes:
            return None
        return np.stack(dose_planes, axis=0)

    def build_partitioned_stereotactic_context(
        self,
        structure: StructureSliceContours,
        source_key: str,
        *,
        margin_mm: float,
    ) -> Optional[Dict[str, object]]:
        if self.ct is None or self.rtstruct is None:
            return None

        try:
            from scipy.ndimage import distance_transform_edt  # type: ignore
        except Exception:  # pragma: no cover - optional runtime dependency
            return None

        normalized_name = normalize_structure_name(structure.name)
        target_masks = self.get_target_structure_slice_masks(structure)
        if not target_masks:
            return None

        sx = float(self.ct.spacing_xyz_mm[0])
        sy = float(self.ct.spacing_xyz_mm[1])
        sz = float(self.ct.spacing_xyz_mm[2])
        voxel_volume_cc = sx * sy * sz / 1000.0

        def expanded_bbox_from_masks(mask_map: Dict[int, np.ndarray]) -> Optional[Tuple[int, int, int, int, int, int]]:
            if not mask_map:
                return None
            slice_indices = sorted(mask_map)
            row_min: Optional[int] = None
            row_max: Optional[int] = None
            col_min: Optional[int] = None
            col_max: Optional[int] = None
            for slice_index, mask in mask_map.items():
                coords = np.argwhere(mask)
                if coords.size == 0:
                    continue
                local_row_min = int(np.min(coords[:, 0]))
                local_row_max = int(np.max(coords[:, 0]))
                local_col_min = int(np.min(coords[:, 1]))
                local_col_max = int(np.max(coords[:, 1]))
                row_min = local_row_min if row_min is None else min(row_min, local_row_min)
                row_max = local_row_max if row_max is None else max(row_max, local_row_max)
                col_min = local_col_min if col_min is None else min(col_min, local_col_min)
                col_max = local_col_max if col_max is None else max(col_max, local_col_max)
            if row_min is None or row_max is None or col_min is None or col_max is None:
                return None
            margin_rows = int(np.ceil(margin_mm / max(sy, 1e-6)))
            margin_cols = int(np.ceil(margin_mm / max(sx, 1e-6)))
            margin_slices = int(np.ceil(margin_mm / max(sz, 1e-6)))
            z_start = max(0, slice_indices[0] - margin_slices)
            z_end = min(self.ct.volume_hu.shape[0] - 1, slice_indices[-1] + margin_slices)
            return (
                z_start,
                z_end,
                max(0, row_min - margin_rows),
                min(self.ct.rows - 1, row_max + margin_rows),
                max(0, col_min - margin_cols),
                min(self.ct.cols - 1, col_max + margin_cols),
            )

        target_bbox = expanded_bbox_from_masks(target_masks)
        if target_bbox is None:
            return None

        def boxes_intersect(
            box_a: Tuple[int, int, int, int, int, int],
            box_b: Tuple[int, int, int, int, int, int],
        ) -> bool:
            return not (
                box_a[1] < box_b[0]
                or box_b[1] < box_a[0]
                or box_a[3] < box_b[2]
                or box_b[3] < box_a[2]
                or box_a[5] < box_b[4]
                or box_b[5] < box_a[4]
            )

        relevant_ptv_entries: List[Tuple[str, Dict[int, np.ndarray], Tuple[int, int, int, int, int, int]]] = []
        for ptv_normalized_name, ptv_masks in self.get_stereotactic_competing_ptv_entries(structure):
            ptv_bbox = expanded_bbox_from_masks(ptv_masks)
            if ptv_bbox is None:
                continue
            if not boxes_intersect(target_bbox, ptv_bbox):
                continue
            relevant_ptv_entries.append((ptv_normalized_name, ptv_masks, ptv_bbox))

        if not relevant_ptv_entries:
            return None

        z_start = min(entry[2][0] for entry in relevant_ptv_entries)
        z_end = max(entry[2][1] for entry in relevant_ptv_entries)
        row_start = min(entry[2][2] for entry in relevant_ptv_entries)
        row_end = max(entry[2][3] for entry in relevant_ptv_entries)
        col_start = min(entry[2][4] for entry in relevant_ptv_entries)
        col_end = max(entry[2][5] for entry in relevant_ptv_entries)

        local_shape = (
            z_end - z_start + 1,
            row_end - row_start + 1,
            col_end - col_start + 1,
        )
        local_ptv_masks: List[np.ndarray] = []
        target_index: Optional[int] = None
        for index, (ptv_normalized_name, ptv_masks, _ptv_bbox) in enumerate(relevant_ptv_entries):
            local_mask = np.zeros(local_shape, dtype=bool)
            for slice_index, slice_mask in ptv_masks.items():
                if slice_index < z_start or slice_index > z_end:
                    continue
                local_z = slice_index - z_start
                local_mask[local_z] = slice_mask[row_start: row_end + 1, col_start: col_end + 1]
            local_ptv_masks.append(local_mask)
            if ptv_normalized_name == normalized_name:
                target_index = index

        if target_index is None:
            return None

        dose_block = self.get_local_target_dose_block(
            source_key,
            z_start,
            z_end,
            row_start,
            row_end,
            col_start,
            col_end,
        )
        if dose_block is None or dose_block.shape != local_shape:
            return None

        sampling = (sz, sy, sx)
        outside_distance_maps: List[np.ndarray] = []
        surface_distance_maps: List[np.ndarray] = []
        for local_mask in local_ptv_masks:
            outside_distance = distance_transform_edt(~local_mask, sampling=sampling).astype(np.float32, copy=False)
            inside_distance = distance_transform_edt(local_mask, sampling=sampling).astype(np.float32, copy=False)
            surface_distance = np.where(local_mask, inside_distance, outside_distance).astype(np.float32, copy=False)
            outside_distance_maps.append(outside_distance)
            surface_distance_maps.append(surface_distance)

        target_outside_distance = outside_distance_maps[target_index]
        target_expanded_mask = target_outside_distance <= margin_mm
        if not np.any(target_expanded_mask):
            return None

        eligible_distance_stack = np.stack(
            [
                np.where(outside_distance <= margin_mm, surface_distance, np.inf)
                for outside_distance, surface_distance in zip(outside_distance_maps, surface_distance_maps)
            ],
            axis=0,
        )
        min_distance = np.min(eligible_distance_stack, axis=0)
        finite_mask = np.isfinite(min_distance) & target_expanded_mask
        if not np.any(finite_mask):
            return None

        tie_mask = np.isclose(
            eligible_distance_stack,
            min_distance[None, ...],
            atol=1e-6,
        ) & np.isfinite(eligible_distance_stack)
        tie_count = np.sum(tie_mask, axis=0)
        target_weight = np.where(
            finite_mask & tie_mask[target_index],
            1.0 / np.maximum(tie_count, 1),
            0.0,
        ).astype(np.float32, copy=False)
        return {
            "target_weight": target_weight,
            "dose_block": dose_block,
            "voxel_volume_cc": voxel_volume_cc,
            "relevant_ptv_count": len(relevant_ptv_entries),
            "z_start": z_start,
            "z_end": z_end,
            "row_start": row_start,
            "row_end": row_end,
            "col_start": col_start,
            "col_end": col_end,
        }

    def compute_partitioned_stereotactic_volume_cc(
        self,
        context: Dict[str, object],
        threshold_gy: float,
        extra_mask: Optional[np.ndarray] = None,
    ) -> float:
        if threshold_gy <= 0.0:
            return 0.0

        target_weight = np.asarray(context["target_weight"], dtype=np.float32)
        dose_block = np.asarray(context["dose_block"], dtype=np.float32)
        voxel_volume_cc = float(context["voxel_volume_cc"])

        isodose_mask = np.asarray(dose_block >= float(threshold_gy), dtype=bool)
        for z_index in range(isodose_mask.shape[0]):
            if np.any(isodose_mask[z_index]):
                isodose_mask[z_index] = fill_binary_holes_2d(isodose_mask[z_index])

        if extra_mask is not None:
            isodose_mask &= np.asarray(extra_mask, dtype=bool)

        return float(np.sum(target_weight[isodose_mask]) * voxel_volume_cc)

    def compute_partitioned_nearest_ptv_threshold_volume_cc(
        self,
        structure: StructureSliceContours,
        source_key: str,
        threshold_gy: float,
        extra_mask: Optional[np.ndarray] = None,
    ) -> float:
        if self.ct is None or self.rtstruct is None or threshold_gy <= 0.0:
            return 0.0

        try:
            from scipy.ndimage import distance_transform_edt  # type: ignore
        except Exception:  # pragma: no cover - optional runtime dependency
            return 0.0

        dose_volume_ct = self.get_target_dose_volume_ct(source_key)
        if dose_volume_ct is None:
            return 0.0

        isodose_mask = np.asarray(dose_volume_ct >= float(threshold_gy), dtype=bool)
        if extra_mask is not None:
            isodose_mask &= np.asarray(extra_mask, dtype=bool)
        if not np.any(isodose_mask):
            return 0.0

        relevant_ptv_entries = self.get_stereotactic_competing_ptv_entries(structure)
        if not relevant_ptv_entries:
            return 0.0

        coords = np.argwhere(isodose_mask)
        z_start = int(np.min(coords[:, 0]))
        z_end = int(np.max(coords[:, 0]))
        row_start = int(np.min(coords[:, 1]))
        row_end = int(np.max(coords[:, 1]))
        col_start = int(np.min(coords[:, 2]))
        col_end = int(np.max(coords[:, 2]))

        for _ptv_name, ptv_masks in relevant_ptv_entries:
            slice_indices = sorted(ptv_masks)
            if slice_indices:
                z_start = min(z_start, slice_indices[0])
                z_end = max(z_end, slice_indices[-1])
            for slice_mask in ptv_masks.values():
                coords_2d = np.argwhere(slice_mask)
                if coords_2d.size == 0:
                    continue
                row_start = min(row_start, int(np.min(coords_2d[:, 0])))
                row_end = max(row_end, int(np.max(coords_2d[:, 0])))
                col_start = min(col_start, int(np.min(coords_2d[:, 1])))
                col_end = max(col_end, int(np.max(coords_2d[:, 1])))

        local_threshold_mask = isodose_mask[
            z_start: z_end + 1,
            row_start: row_end + 1,
            col_start: col_end + 1,
        ]
        if not np.any(local_threshold_mask):
            return 0.0

        local_shape = local_threshold_mask.shape
        local_ptv_masks: List[np.ndarray] = []
        target_index: Optional[int] = None
        target_normalized_name = normalize_structure_name(structure.name)
        for index, (ptv_normalized_name, ptv_masks) in enumerate(relevant_ptv_entries):
            local_mask = np.zeros(local_shape, dtype=bool)
            for slice_index, slice_mask in ptv_masks.items():
                if slice_index < z_start or slice_index > z_end:
                    continue
                local_z = slice_index - z_start
                local_mask[local_z] = slice_mask[row_start: row_end + 1, col_start: col_end + 1]
            local_ptv_masks.append(local_mask)
            if ptv_normalized_name == target_normalized_name:
                target_index = index

        if target_index is None:
            return 0.0

        voxel_volume_cc = float(np.prod(self.ct.spacing_xyz_mm) / 1000.0)
        if len(local_ptv_masks) == 1:
            return float(np.count_nonzero(local_threshold_mask)) * voxel_volume_cc

        sampling = (
            float(self.ct.spacing_xyz_mm[2]),
            float(self.ct.spacing_xyz_mm[1]),
            float(self.ct.spacing_xyz_mm[0]),
        )
        eligible_distance_stack = np.stack(
            [
                np.where(
                    local_threshold_mask,
                    np.where(
                        local_mask,
                        distance_transform_edt(local_mask, sampling=sampling),
                        distance_transform_edt(~local_mask, sampling=sampling),
                    ),
                    np.inf,
                )
                for local_mask in local_ptv_masks
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        min_distance = np.min(eligible_distance_stack, axis=0)
        finite_mask = np.isfinite(min_distance) & local_threshold_mask
        if not np.any(finite_mask):
            return 0.0

        tie_mask = np.isclose(
            eligible_distance_stack,
            min_distance[None, ...],
            atol=1e-6,
        ) & np.isfinite(eligible_distance_stack)
        tie_count = np.sum(tie_mask, axis=0)
        target_weight = np.where(
            finite_mask & tie_mask[target_index],
            1.0 / np.maximum(tie_count, 1),
            0.0,
        ).astype(np.float32, copy=False)
        return float(np.sum(target_weight[local_threshold_mask]) * voxel_volume_cc)

    def get_target_fraction_count(
        self,
        normalized_name: str,
        source_key: str,
        phase_assignments: Dict[str, Tuple[RTPlanPhase, float]],
        single_phase: Optional[RTPlanPhase],
    ) -> int:
        return get_target_fraction_count_helper(
            normalized_name,
            source_key,
            phase_assignments=phase_assignments,
            single_phase=single_phase,
            plan_phases=self.plan_phases,
        )

    def compute_stereotactic_indices(
        self,
        structure: StructureSliceContours,
        threshold_gy: float,
        coverage_pct: float,
        source_key: str,
        fractions_planned: int,
    ) -> Tuple[str, str, str, str, str]:
        if self.ct is None:
            return "", "", "", "", ""
        return compute_stereotactic_indices_helper(
            structure,
            threshold_gy,
            coverage_pct,
            source_key,
            fractions_planned,
            stereotactic_metrics_cache=self.stereotactic_metrics_cache,
            get_structure_geometry_volume_cc=self.get_structure_geometry_volume_cc,
            get_stereotactic_volume_context=lambda target_structure, target_source_key, minimum_threshold: self.get_stereotactic_volume_context(
                target_structure,
                target_source_key,
                minimum_threshold,
            ),
            compute_stereotactic_owned_volume_cc=lambda context, value_threshold_gy, extra_mask: self.compute_stereotactic_owned_volume_cc(
                context,
                value_threshold_gy,
                extra_mask=extra_mask,
            ),
            get_brain_structure=self.get_brain_structure,
            get_structure_volume_mask=self.get_structure_volume_mask,
            get_nested_target_structures=self.get_nested_target_structures,
        )

    def get_primary_target_context(
        self,
        structure: StructureSliceContours,
        phase_assignments: Dict[str, Tuple[RTPlanPhase, float]],
        single_phase: Optional[RTPlanPhase],
    ) -> Tuple[float, str, Tuple[Optional[float], Optional[float], Optional[float]]]:
        return get_primary_target_context_helper(
            structure,
            phase_assignments=phase_assignments,
            single_phase=single_phase,
            parse_ptv_rx_gy_from_name=self.parse_ptv_rx_gy_from_name,
            get_stereotactic_threshold_gy=self.get_stereotactic_threshold_gy,
            compute_structure_target_metric_values=lambda target_structure, threshold_gy, target_source_key: self.compute_structure_target_metric_values(
                target_structure,
                threshold_gy,
                source_key=target_source_key,
            ),
            has_sampled_dose_volume_ct=self.sampled_dose_volume_ct is not None,
        )

    def get_phase_target_assignments(self) -> Dict[str, Tuple[RTPlanPhase, float]]:
        return get_phase_target_assignments_helper(
            self.plan_phases,
            self.get_sorted_ptv_structures(),
            parse_ptv_rx_gy_from_name=self.parse_ptv_rx_gy_from_name,
        )

    def build_target_table_rows(self) -> List[Dict[str, object]]:
        return build_target_table_rows_helper(
            rtstruct=self.rtstruct,
            has_ct=self.ct is not None,
            plan_phases=self.plan_phases,
            sorted_ptv_structures=self.get_sorted_ptv_structures(),
            phase_assignments=self.get_phase_target_assignments(),
            stereotactic_summary_enabled=self.stereotactic_summary_enabled(),
            parse_ptv_rx_gy_from_name=self.parse_ptv_rx_gy_from_name,
            get_stereotactic_threshold_gy=self.get_stereotactic_threshold_gy,
            get_primary_target_context=self.get_primary_target_context,
            get_target_fraction_count=self.get_target_fraction_count,
            compute_stereotactic_indices=self.compute_stereotactic_indices,
            get_nested_target_structures=self.get_nested_target_structures,
            compute_structure_target_metric_values=lambda target_structure, threshold_gy, target_source_key: self.compute_structure_target_metric_values(
                target_structure,
                threshold_gy,
                source_key=target_source_key,
            ),
            compute_structure_target_metrics=lambda target_structure, threshold_gy, target_source_key: self.compute_structure_target_metrics(
                target_structure,
                threshold_gy,
                source_key=target_source_key,
            ),
            format_target_dose_text=self.format_target_dose_text,
        )

    def get_target_table_rows(self) -> List[Dict[str, object]]:
        if self.cached_target_table_rows is None:
            if self.restore_saved_results_without_calculation:
                return []
            progress_message = "Computing SRS metrics" if self.stereotactic_summary_enabled() else "Computing metrics"
            self.show_progress_status(progress_message)
            self.cached_target_table_rows = self.build_target_table_rows()
            self.clear_progress_status(progress_message)
        return self.cached_target_table_rows

    def update_targets_table_column_widths(self):
        if self.targets_table.columnCount() != 6:
            return

        column_widths = get_target_table_column_widths(self.targets_table.viewport().width())

        for column_index, width in enumerate(column_widths):
            self.targets_table.setColumnWidth(column_index, max(1, width))
        self.targets_table.resizeRowsToContents()

    def create_target_name_cell_widget(
        self,
        display_name: str,
        color_rgb: Tuple[int, int, int],
        background_color: QtGui.QColor,
        *,
        structure_name: Optional[str] = None,
        normalized_name: Optional[str] = None,
        is_primary_ptv: bool = False,
    ) -> QtWidgets.QWidget:
        return create_target_name_cell_widget_helper(
            display_name,
            color_rgb,
            background_color,
            font_point_size=self.targets_table.font().pointSize(),
            is_primary_ptv=is_primary_ptv,
        )

    def create_target_coverage_cell_widget(
        self,
        coverage_text: str,
        background_color: QtGui.QColor,
        *,
        structure_name: str,
        normalized_name: str,
        is_primary_ptv: bool,
        resolved_dose_text: Optional[str] = None,
    ) -> QtWidgets.QWidget:
        return create_target_coverage_cell_widget_helper(
            coverage_text,
            background_color,
            structure_name=structure_name,
            normalized_name=normalized_name,
            is_primary_ptv=is_primary_ptv,
            resolved_dose_text=resolved_dose_text,
            fallback_dose_text=self.get_stereotactic_dose_text(normalized_name, structure_name),
            font_point_size=self.targets_table.font().pointSize(),
            on_editing_finished=self.on_stereotactic_dose_editing_finished,
        )

    def on_stereotactic_dose_editing_finished(
        self,
        normalized_name: str,
        structure_name: str,
        edit: QtWidgets.QLineEdit,
    ):
        dose_text = self.normalize_stereotactic_dose_text(edit.text())
        default_text = self.normalize_stereotactic_dose_text(self.get_default_stereotactic_dose_text(structure_name))
        stored_text = self.normalize_stereotactic_dose_text(
            self.stereotactic_target_dose_text_by_name.get(normalized_name, "")
        )

        if dose_text == default_text:
            if stored_text:
                self.stereotactic_target_dose_text_by_name.pop(normalized_name, None)
                if stored_text == default_text:
                    edit.setText(default_text)
                    return
            else:
                edit.setText(default_text)
                return
        elif dose_text == stored_text:
            edit.setText(dose_text)
            return
        elif not dose_text:
            if not stored_text:
                edit.setText(default_text)
                return
            self.stereotactic_target_dose_text_by_name.pop(normalized_name, None)
            dose_text = default_text
        else:
            self.stereotactic_target_dose_text_by_name[normalized_name] = dose_text

        if dose_text:
            edit.setText(dose_text)
        else:
            edit.clear()
        self.stereotactic_metrics_cache = {}
        self.stereotactic_volume_context_cache = {}
        self.cached_target_table_rows = None
        self.update_targets_table()

    def update_constraints_table(self):
        if self.rtstruct is None:
            self.constraints_table.setRowCount(0)
            self.constraint_editor_widgets = {}
            self.constraints_table_refresh_pending = False
            return

        if self.tabs.currentWidget() is not self.constraints_tab:
            self.constraints_table_refresh_pending = True
            return

        self.constraints_table_refresh_pending = False
        rows = build_constraints_table_presentation_rows(
            rtstruct=self.rtstruct,
            structure_goals_by_name=self.structure_goals_by_name,
            structure_goal_evaluations=self.structure_goal_evaluations,
            dvh_structure_goal_evaluation_cache=self.dvh_structure_goal_evaluation_cache,
            constraint_notes=self.constraint_notes,
            constraints_sheet_name=self.constraints_sheet_name or "",
            get_curve_for_name=self.get_curve_for_name,
            get_constraint_note_key=self.get_constraint_note_key,
            is_custom_only_constraint=self.is_custom_only_constraint,
            allow_curve_computation=not self.restore_saved_results_without_calculation,
        )
        self.constraint_editor_widgets = refresh_constraints_table(
            self.constraints_table,
            rows,
            constraint_editor_state=self.constraint_editor_state,
            structure_names=self.get_listable_structure_names(),
            on_edit_note=self.edit_constraint_note,
            on_field_change=self.on_constraint_editor_field_changed,
            on_commit=self.commit_constraint_editor,
            on_cancel=self.cancel_constraint_editor,
        )
        if self.constraint_editor_state is not None:
            self.update_constraint_editor_preview()

        self.constraints_table.resizeColumnsToContents()
        QtCore.QTimer.singleShot(0, self.update_constraints_table_column_widths)

    def get_constraint_evaluations_for_structure(
        self,
        normalized_name: str,
        goals: List[StructureGoal],
    ) -> List[StructureGoalEvaluation]:
        return get_constraint_evaluations_for_structure_helper(
            normalized_name,
            goals,
            structure_goal_evaluations=self.structure_goal_evaluations,
            dvh_structure_goal_evaluation_cache=self.dvh_structure_goal_evaluation_cache,
            get_curve_for_name=self.get_curve_for_name,
            allow_curve_computation=not self.restore_saved_results_without_calculation,
        )

    def compose_constraint_note_text(self, computed_note_text: str, stored_note_text: str) -> str:
        return compose_constraint_note_text_helper(computed_note_text, stored_note_text)

    def prostate_constraint_summary_enabled(self) -> bool:
        return prostate_constraint_summary_enabled_helper(self.constraints_sheet_name or "")

    def get_min_bladder_volume_note_text(self) -> str:
        return get_min_bladder_volume_note_text_helper(
            constraints_sheet_name=self.constraints_sheet_name or "",
            structure_goals_by_name=self.structure_goals_by_name,
            get_curve_for_name=self.get_curve_for_name,
            allow_curve_computation=not self.restore_saved_results_without_calculation,
        )

    def get_computed_constraint_note_text(
        self,
        normalized_name: str,
        goals: List[StructureGoal],
        goal_index: int,
        evaluation: Optional[StructureGoalEvaluation] = None,
    ) -> str:
        return get_computed_constraint_note_text_helper(
            normalized_name,
            goals,
            goal_index,
            evaluation=evaluation,
            constraints_sheet_name=self.constraints_sheet_name or "",
            structure_goals_by_name=self.structure_goals_by_name,
            get_curve_for_name=self.get_curve_for_name,
            allow_curve_computation=not self.restore_saved_results_without_calculation,
        )

    def update_targets_table(self):
        if self.rtstruct is not None and self.ct is not None and self.tabs.currentWidget() is not self.targets_tab:
            self.targets_table_refresh_pending = True
            return

        self.targets_table_refresh_pending = False
        self.targets_table.setUpdatesEnabled(False)
        self.targets_table.setRowCount(0)

        if self.rtstruct is None or self.ct is None:
            self.targets_table.setUpdatesEnabled(True)
            return

        rows = self.get_target_table_rows()
        presentation_rows = build_target_table_presentation_rows(
            rows,
            target_notes=self.target_notes,
            get_target_note_key_for_row=self.get_target_note_key_for_row,
            compose_target_note_text=self.compose_target_note_text,
            get_target_row_reference_dose_text=self.get_target_row_reference_dose_text,
        )
        primary_background = QtGui.QColor(8, 8, 8)
        nested_background = QtGui.QColor(34, 34, 34)
        populate_target_table_rows(
            self.targets_table,
            presentation_rows,
            primary_background=primary_background,
            nested_background=nested_background,
            get_fallback_dose_text=self.get_stereotactic_dose_text,
            on_editing_finished=self.on_stereotactic_dose_editing_finished,
            on_edit_note=self.edit_target_note,
        )
        self.targets_table.setUpdatesEnabled(True)
        self.targets_table.viewport().update()
        QtCore.QTimer.singleShot(0, self.update_targets_table_column_widths)
        QtCore.QTimer.singleShot(0, self.targets_table.resizeRowsToContents)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self.update_constraints_table_column_widths()
        self.update_targets_table_column_widths()
        self.update_axial_overlay_positions()

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.stop_patient_transition_overlay_process()
        super().closeEvent(event)

    def get_structure_goal_lines(self, normalized_name: str) -> List[Tuple[str, Optional[str]]]:
        goals = self.structure_goals_by_name.get(normalized_name, [])
        if not goals:
            return []
        return [
            (
                self.format_structure_goal_line(evaluation),
                self.structure_goal_line_color(evaluation),
            )
            for evaluation in self.get_constraint_evaluations_for_structure(normalized_name, goals)
        ]

    def get_dvh_structure_goal_lines(self, normalized_name: str) -> List[Tuple[str, Optional[str]]]:
        lines: List[Tuple[str, Optional[str]]] = []
        if normalized_name.startswith("PTV"):
            coverage_text = self.get_cached_ptv_coverage_text(normalized_name)
            if coverage_text:
                lines.append((coverage_text, None))
        lines.extend(
            (
                self.format_structure_goal_line(evaluation),
                self.structure_goal_line_color(evaluation),
            )
            for evaluation in self.dvh_structure_goal_evaluation_cache.get(normalized_name, [])
        )
        return lines

    def get_dvh_structure_secondary_text(self, normalized_name: str) -> Tuple[Optional[str], Optional[str]]:
        parts: List[str] = []

        volume_cc = self.dvh_structure_volume_cache.get(normalized_name)
        if volume_cc is None:
            structure = self.get_structure_by_normalized_name(normalized_name)
            if structure is not None and not self.restore_saved_results_without_calculation:
                volume_cc = self.get_structure_geometry_volume_cc(structure)
                self.dvh_structure_volume_cache[normalized_name] = volume_cc
            else:
                curve = self.get_curve_for_name(normalized_name)
                if curve is not None:
                    volume_cc = float(curve.volume_cc)
        if volume_cc is not None:
            parts.append(f"Vol {volume_cc:.2f} cc")

        if not parts:
            return None, None
        return "   ".join(parts), "#d0d0d0"

    def get_dvh_structure_item_options(self, normalized_name: str) -> Dict[str, object]:
        secondary_text, secondary_text_color = self.get_dvh_structure_secondary_text(normalized_name)
        return {
            "secondary_text": secondary_text,
            "secondary_text_color": secondary_text_color,
        }

    def get_structure_by_normalized_name(self, normalized_name: str) -> Optional[StructureSliceContours]:
        if self.rtstruct is None:
            return None
        for structure in self.rtstruct.structures:
            if normalize_structure_name(structure.name) == normalized_name:
                return structure
        return None

    def get_ptv_coverage_goal_lines(self, normalized_name: str) -> List[Tuple[str, Optional[str]]]:
        if self.ct is None:
            return []

        cached_coverage_text = self.get_cached_ptv_coverage_text(normalized_name)
        if cached_coverage_text:
            return [(f"Coverage {cached_coverage_text}", None)]

        if self.defer_sidebar_summary_metrics or self.restore_saved_results_without_calculation:
            return []

        structure = self.get_structure_by_normalized_name(normalized_name)
        if structure is None:
            return []

        total_rx_gy = self.parse_ptv_rx_gy_from_name(structure.name)
        if total_rx_gy is None:
            return []

        phase_assignment = self.get_phase_target_assignments().get(normalized_name)
        if phase_assignment is not None:
            phase, phase_rx_gy = phase_assignment
            _minimum_dose_text, _maximum_dose_text, coverage_text = self.compute_structure_target_metrics(
                structure,
                phase_rx_gy,
                source_key=phase.dose_path,
            )
        elif self.sampled_dose_volume_ct is not None:
            _minimum_dose_text, _maximum_dose_text, coverage_text = self.compute_structure_target_metrics(
                structure,
                total_rx_gy,
                source_key="combined",
            )
        else:
            return []

        return [(f"Coverage {coverage_text}", None)]

    def get_max_tissue_dose_goal_lines(self) -> List[Tuple[str, Optional[str]]]:
        if self.ct is None or self.rtstruct is None or self.sampled_dose_volume_ct is None:
            return []

        if self.max_tissue_dose_gy_cache is None and (
            self.defer_sidebar_summary_metrics or self.restore_saved_results_without_calculation
        ):
            return []

        if self.max_tissue_dose_gy_cache is None:
            ptv_union_masks = self.get_ptv_union_slice_masks()
            if not ptv_union_masks and not any(
                normalize_structure_name(structure.name).startswith("PTV")
                for structure in self.rtstruct.structures
            ):
                return []

            tissue_mask = np.isfinite(self.sampled_dose_volume_ct)
            ptv_union_volume_mask = self.get_ptv_union_volume_mask()
            if ptv_union_volume_mask is not None and ptv_union_volume_mask.shape == tissue_mask.shape:
                tissue_mask &= ~ptv_union_volume_mask
            if np.any(tissue_mask):
                tissue_dose_volume = np.where(tissue_mask, self.sampled_dose_volume_ct, -np.inf)
                max_tissue_dose_gy = float(np.max(tissue_dose_volume))
                if np.isfinite(max_tissue_dose_gy):
                    flat_index = int(np.argmax(tissue_dose_volume))
                    max_index = np.unravel_index(flat_index, tissue_dose_volume.shape)
                    self.max_tissue_dose_gy_cache = max_tissue_dose_gy
                    self.max_tissue_index_zyx = (
                        int(max_index[0]),
                        int(max_index[1]),
                        int(max_index[2]),
                    )
                else:
                    self.max_tissue_dose_gy_cache = None
                    self.max_tissue_index_zyx = None
            else:
                self.max_tissue_dose_gy_cache = None
                self.max_tissue_index_zyx = None

        if self.max_tissue_dose_gy_cache is None:
            return []
        return [(f"{self.max_tissue_dose_gy_cache:.2f} Gy", None)]

    def get_axial_structure_goal_lines(self, normalized_name: str) -> List[Tuple[str, Optional[str]]]:
        if normalized_name == MAX_TISSUE_ROW_NAME:
            return self.get_max_tissue_dose_goal_lines()
        lines: List[Tuple[str, Optional[str]]] = []
        if normalized_name.startswith("PTV"):
            lines.extend(self.get_ptv_coverage_goal_lines(normalized_name))
        lines.extend(self.get_structure_goal_lines(normalized_name))
        return lines

    def update_structure_list_goal_texts(self):
        self.axial_structure_list.update_goal_lines(self.build_axial_list_rtstruct(), self.get_axial_structure_goal_lines)
        self.dvh_structure_list.update_goal_lines(self.rtstruct, self.get_dvh_structure_goal_lines)
        self.dvh_structure_list.update_secondary_texts(self.rtstruct, self.get_dvh_structure_secondary_text)
        self.update_constraints_table()
        self.update_targets_table()

    def is_base_listable_structure_name(self, normalized_name: str) -> bool:
        excluded_fragments = ("COUCH", "RAIL", "BB")
        return not normalized_name.startswith("Z") and not any(
            fragment in normalized_name for fragment in excluded_fragments
        )

    def is_listable_structure_name(self, normalized_name: str) -> bool:
        return self.is_base_listable_structure_name(normalized_name) and normalized_name not in self.hidden_structure_names

    def build_listable_rtstruct(self) -> Optional[RTStructData]:
        if self.rtstruct is None:
            return None

        structures = [
            structure
            for structure in self.rtstruct.structures
            if self.is_listable_structure_name(normalize_structure_name(structure.name))
        ]
        return RTStructData(
            structures=structures,
            frame_of_reference_uid=self.rtstruct.frame_of_reference_uid,
        )

    def build_axial_list_rtstruct(self, *, include_max_tissue: bool = True) -> Optional[RTStructData]:
        listable_rtstruct = self.build_listable_rtstruct()
        if listable_rtstruct is None:
            return None

        ptv_structures = [
            structure
            for structure in listable_rtstruct.structures
            if normalize_structure_name(structure.name).startswith("PTV")
        ]
        non_ptv_structures = [
            structure
            for structure in listable_rtstruct.structures
            if not normalize_structure_name(structure.name).startswith("PTV")
        ]

        structures: List[StructureSliceContours] = list(ptv_structures)
        if include_max_tissue and self.get_max_tissue_dose_goal_lines():
            structures.append(
                StructureSliceContours(
                    name=MAX_TISSUE_ROW_LABEL,
                    color_rgb=(255, 255, 255),
                    points_rc_by_slice={},
                )
            )
        structures.extend(non_ptv_structures)
        return RTStructData(
            structures=structures,
            frame_of_reference_uid=listable_rtstruct.frame_of_reference_uid,
        )

    def populate_structure_sidebars(self, *, quick_mode: bool = False) -> None:
        axial_list_rtstruct = self.build_axial_list_rtstruct(include_max_tissue=not quick_mode)
        listable_rtstruct = self.build_listable_rtstruct()
        axial_goal_lines_getter = self.get_axial_structure_goal_lines if not quick_mode else (lambda _normalized_name: [])
        dvh_goal_lines_getter = self.get_dvh_structure_goal_lines if not quick_mode else (lambda _normalized_name: [])
        dvh_item_options_getter = self.get_dvh_structure_item_options if not quick_mode else (lambda _normalized_name: {})
        self.axial_structure_list.set_structures(
            axial_list_rtstruct,
            axial_goal_lines_getter,
            default_visibility_resolver=lambda normalized_name: normalized_name.startswith("PTV"),
            show_checkbox_resolver=lambda normalized_name: normalized_name != MAX_TISSUE_ROW_NAME,
            item_options_getter=lambda normalized_name: (
                {
                    "trailing_button_text": "MT",
                    "trailing_button_callback": self.on_go_to_max_tissue,
                    "inline_goals": True,
                    "inline_goals_compact": True,
                }
                if normalized_name == MAX_TISSUE_ROW_NAME
                else {}
            ),
        )
        self.dvh_structure_list.set_structures(
            listable_rtstruct,
            dvh_goal_lines_getter,
            default_visibility_resolver=lambda normalized_name: (
                normalized_name.startswith("PTV")
                or normalized_name in self.structure_goals_by_name
            ),
            item_options_getter=dvh_item_options_getter,
        )
        if self.pending_saved_dvh_selected_names:
            self.dvh_structure_list.set_checked_names(self.pending_saved_dvh_selected_names)
            self.pending_saved_dvh_selected_names = None

    def populate_structures_list(self):
        self.populate_structure_sidebars()
        self.update_constraints_table()
        self.update_targets_table()

    def get_selected_dvh_structure_names(self) -> List[str]:
        return get_selected_dvh_structure_names_helper(self.rtstruct, self.dvh_structure_is_visible)

    def build_selected_dvh_rtstruct(self, selected_names: Optional[List[str]] = None) -> Optional[RTStructData]:
        return build_selected_dvh_rtstruct_helper(
            self.rtstruct,
            selected_names if selected_names is not None else self.get_selected_dvh_structure_names(),
        )

    def structure_is_visible(self, idx: int) -> bool:
        if self.rtstruct is None or idx >= len(self.rtstruct.structures):
            return True
        normalized_name = normalize_structure_name(self.rtstruct.structures[idx].name)
        if not self.is_listable_structure_name(normalized_name):
            return False
        return self.axial_structure_list.is_visible(normalized_name)

    def dvh_structure_is_visible(self, normalized_name: str) -> bool:
        if not self.is_listable_structure_name(normalized_name):
            return False
        return self.dvh_structure_list.is_visible(normalized_name)

    def wheelEvent(self, event):
        if self.ct is None:
            super().wheelEvent(event)
            return

        if self.autoscroll_button.isChecked():
            event.accept()
            return

        target_widget = self.childAt(event.position().toPoint())
        while target_widget is not None:
            if target_widget in {
                self.structures_list,
                self.structures_list.viewport(),
                self.dvh_structures_list,
                self.dvh_structures_list.viewport(),
                self.constraints_table,
                self.constraints_table.viewport(),
                self.targets_table,
                self.targets_table.viewport(),
            }:
                event.accept()
                return
            target_widget = target_widget.parentWidget()

        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return

        step = 1 if delta > 0 else -1
        self.step_slice(step)
        event.accept()

    def get_window_level(self) -> Tuple[float, float]:
        lower, center, upper = self.window_level_slider.values()
        level = float(center)
        width = float(max(1, upper - lower))
        return width, level

    def clear_overlay_items(self, view: pg.ViewBox, items: List[pg.GraphicsObject]):
        clear_overlay_items_helper(view, items)

    def set_isodose_color_palette(self, colors: Sequence[Tuple[int, int, int]]) -> None:
        if not colors:
            return
        normalized_colors = [tuple(int(component) for component in color[:3]) for color in colors if len(color) >= 3]
        if not normalized_colors:
            return
        while len(normalized_colors) < len(self.isodose_colors):
            normalized_colors.append(self.isodose_colors[len(normalized_colors)])
        self.isodose_colors = normalized_colors[: len(self.isodose_colors)]
        for swatch, color in zip(self.isodose_swatch_widgets, self.isodose_colors):
            swatch.color_rgb = color
            swatch.update()

    def get_default_isodose_level_texts(self) -> List[str]:
        default_levels: List[float] = []
        total_rx_gy = self.get_total_rx_dose_gy()
        if total_rx_gy is not None and total_rx_gy > 0.0:
            default_levels.append(float(total_rx_gy))

        for ptv_level_gy in sorted(get_ptv_dose_levels_gy(self.rtstruct), reverse=True):
            if any(abs(ptv_level_gy - existing_level) < 0.05 for existing_level in default_levels):
                continue
            default_levels.append(float(ptv_level_gy))

        return [f"{level_gy:.1f}" for level_gy in default_levels[: len(self.isodose_edit_widgets)]]

    def apply_isodose_settings(
        self,
        *,
        level_texts: Optional[Sequence[str]] = None,
        colors: Optional[Sequence[Tuple[int, int, int]]] = None,
    ) -> None:
        if colors is not None:
            self.set_isodose_color_palette(list(colors))

        if level_texts is None:
            level_texts_to_apply = self.get_default_isodose_level_texts()
        else:
            level_texts_to_apply = [str(text).strip() for text in level_texts]

        for idx, edit in enumerate(self.isodose_edit_widgets):
            blocker = QtCore.QSignalBlocker(edit)
            edit.setText(level_texts_to_apply[idx] if idx < len(level_texts_to_apply) else "")
            del blocker

    def populate_isodose_controls(self):
        self.apply_isodose_settings()

    def get_active_isodose_levels(self) -> List[Tuple[float, Tuple[int, int, int]]]:
        return build_active_isodose_levels(
            [edit.text() for edit in self.isodose_edit_widgets],
            list(self.isodose_colors),
        )

    def on_isodose_editing_finished(self):
        sender = self.sender()
        if isinstance(sender, QtWidgets.QLineEdit):
            text = sender.text().strip()
            if text:
                try:
                    sender.setText(f"{float(text):.1f}")
                except ValueError:
                    pass
        self.isodose_refresh_timer.start(0)

    def on_isodose_text_changed(self, text: str):
        if text.strip():
            return
        self.isodose_refresh_timer.start(0)

    def add_isodose_items(
        self,
        view: pg.ViewBox,
        target_items: List[pg.IsocurveItem],
        dose_plane: Optional[np.ndarray],
    ):
        if self.rtstruct is None:
            self.clear_overlay_items(view, target_items)
            return
        apply_isodose_items(view, target_items, dose_plane, self.get_active_isodose_levels())

    def update_max_dose_markers(self):
        marker_state = build_max_dose_marker_state(
            self.ct,
            self.max_dose_index_zyx,
            int(self.slice_slider.value()),
            self.current_row,
            self.current_col,
        )
        if marker_state.axial_point is None:
            self.axial_max_marker.setData([], [])
        else:
            self.axial_max_marker.setData([marker_state.axial_point[0]], [marker_state.axial_point[1]])
        if marker_state.sagittal_point is None:
            self.sagittal_max_marker.setData([], [])
        else:
            self.sagittal_max_marker.setData([marker_state.sagittal_point[0]], [marker_state.sagittal_point[1]])
        if marker_state.coronal_point is None:
            self.coronal_max_marker.setData([], [])
        else:
            self.coronal_max_marker.setData([marker_state.coronal_point[0]], [marker_state.coronal_point[1]])

    def refresh_orthogonal_views_from_controls(self):
        if self.ct is None:
            return
        ww, wl = self.get_window_level()
        lo = wl - ww / 2.0
        hi = wl + ww / 2.0
        self.update_orthogonal_views(lo, hi)

    def refresh_all_views(self):
        self.update_display()
        self.refresh_orthogonal_views_from_controls()

    def update_orthogonal_views(self, lo: float, hi: float):
        if self.ct is None:
            return

        row_idx = int(np.clip(self.current_row, 0, self.ct.rows - 1))
        col_idx = int(np.clip(self.current_col, 0, self.ct.cols - 1))
        min_dose, max_dose = self.get_dose_display_range()
        render_state = build_orthogonal_render_state(
            self.ct,
            self.rtstruct,
            self.sampled_dose_volume_ct,
            row_idx,
            col_idx,
            lo,
            hi,
            self.current_dose_alpha(),
            min_dose,
            max_dose,
            self.structure_is_visible,
        )

        self.sagittal_ct_item.setImage(render_state.sagittal_plane, levels=(lo, hi), autoLevels=False)
        self.coronal_ct_item.setImage(render_state.coronal_plane, levels=(lo, hi), autoLevels=False)
        self.sagittal_dose_item.setImage(render_state.sagittal_dose_rgba, autoLevels=False)
        self.coronal_dose_item.setImage(render_state.coronal_dose_rgba, autoLevels=False)

        if render_state.sagittal_dose_plane is not None and render_state.coronal_dose_plane is not None:
            self.add_isodose_items(self.sagittal_view, self.sagittal_isodose_items, render_state.sagittal_dose_plane)
            self.add_isodose_items(self.coronal_view, self.coronal_isodose_items, render_state.coronal_dose_plane)
        else:
            self.clear_overlay_items(self.sagittal_view, self.sagittal_isodose_items)
            self.clear_overlay_items(self.coronal_view, self.coronal_isodose_items)

        self.clear_overlay_items(self.sagittal_view, self.sagittal_contour_items)
        self.clear_overlay_items(self.coronal_view, self.coronal_contour_items)

        if self.rtstruct is None:
            self.update_max_dose_markers()
            return

        apply_polyline_specs(self.sagittal_view, self.sagittal_contour_items, render_state.sagittal_contours)
        apply_polyline_specs(self.coronal_view, self.coronal_contour_items, render_state.coronal_contours)

        self.update_max_dose_markers()

    def on_axial_clicked(self, event):
        if self.ct is None:
            return

        if self.autoscroll_button.isChecked():
            return

        mouse_point = self.axial_view.mapSceneToView(event.scenePos())
        indices = resolve_axial_indices(self.ct, float(mouse_point.x()), float(mouse_point.y()))
        if indices is None:
            return
        r, c = indices
        self.current_row = r
        self.current_col = c
        ww, wl = self.get_window_level()
        lo = wl - ww / 2.0
        hi = wl + ww / 2.0
        self.update_orthogonal_views(lo, hi)

    def update_display(self):
        if self.ct is None:
            self.displayed_dose_plane = None
            self.axial_readout_label.hide()
            return

        k = int(self.slice_slider.value())
        ww, wl = self.get_window_level()
        lo = wl - ww / 2.0
        hi = wl + ww / 2.0

        min_dose, max_dose = self.get_dose_display_range()
        render_state = build_axial_render_state(
            self.ct,
            self.dose,
            self.rtstruct,
            self.sampled_dose_volume_ct,
            k,
            lo,
            hi,
            self.current_dose_alpha(),
            min_dose,
            max_dose,
            self.structure_is_visible,
        )

        self.ct_item.setImage(render_state.ct_plane, levels=(lo, hi), autoLevels=False)
        self.displayed_dose_plane = render_state.dose_plane
        self.dose_item.setImage(render_state.dose_rgba, autoLevels=False)
        if render_state.dose_plane is not None:
            self.add_isodose_items(self.axial_view, self.axial_isodose_items, render_state.dose_plane)
        else:
            self.clear_overlay_items(self.axial_view, self.axial_isodose_items)

        self.clear_overlay_items(self.axial_view, self.axial_contour_items)
        apply_polyline_specs(self.axial_view, self.axial_contour_items, render_state.contour_specs)

        self.slice_label.setText(render_state.slice_label_text)
        self.z_label.setText(render_state.z_label_text)
        self.window_label.setText(render_state.window_label_text)
        self.update_dose_range_controls()
        self.update_max_dose_markers()
        self.update_axial_overlay_positions()

    def update_axial_overlay_positions(self):
        margin = 10
        if hasattr(self, "axial_autoscroll_overlay"):
            self.axial_autoscroll_overlay.adjustSize()
        if hasattr(self, "axial_readout_label"):
            self.axial_readout_label.adjustSize()
        readout_width = self.axial_readout_label.width() if hasattr(self, "axial_readout_label") else 0
        overlay_positions = build_axial_overlay_positions(
            self.axial_graphics_widget.width(),
            readout_width,
            margin=margin,
        )
        if hasattr(self, "axial_autoscroll_overlay"):
            self.axial_autoscroll_overlay.move(*overlay_positions.autoscroll_pos)
        if not hasattr(self, "axial_readout_label"):
            return
        if self.ct is None:
            self.axial_readout_label.hide()
            return
        self.axial_readout_label.move(*overlay_positions.readout_pos)

    def on_mouse_moved(self, pos):
        if self.ct is None or self.autoscroll_button.isChecked():
            self.axial_readout_label.hide()
            return

        mouse_point = self.axial_view.mapSceneToView(pos)
        indices = resolve_axial_indices(self.ct, float(mouse_point.x()), float(mouse_point.y()))
        if indices is None:
            self.axial_readout_label.hide()
            return
        r, c = indices

        k = int(self.slice_slider.value())
        msg = build_axial_hover_text(self.ct, self.displayed_dose_plane, k, r, c)
        if msg is None:
            self.axial_readout_label.hide()
            return
        self.axial_readout_label.setText(msg)
        self.update_axial_overlay_positions()
        self.axial_readout_label.show()


def main():
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    apply_app_theme(app)
    win = RTPlanReviewWindow()
    win.show()
    sys.exit(app.exec())
