from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

from peer_helpers import (
    build_structure_mask_cache,
    compute_dvh_curves,
    evaluate_structure_goals,
    normalize_structure_name,
)
from peer_models import CTVolume, DVHCurve, DoseVolume, RTStructData, StructureGoal, StructureGoalEvaluation


def build_file_fingerprint(path: Optional[str]) -> Optional[Dict[str, object]]:
    if not path:
        return None
    try:
        stat_result = os.stat(path)
    except OSError:
        return None
    return {
        "name": Path(path).name,
        "size": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }


def build_file_fingerprints(paths: List[str]) -> List[Dict[str, object]]:
    fingerprints: List[Dict[str, object]] = []
    for path in paths:
        fingerprint = build_file_fingerprint(path)
        if fingerprint is not None:
            fingerprints.append(fingerprint)
    return fingerprints


def file_fingerprint_matches(saved_payload: object, current_path: Optional[str]) -> bool:
    if saved_payload is None:
        return False
    if not isinstance(saved_payload, dict):
        return False
    current_fingerprint = build_file_fingerprint(current_path)
    return current_fingerprint is not None and saved_payload == current_fingerprint


def file_fingerprint_list_matches(saved_payload: object, current_paths: List[str]) -> bool:
    if saved_payload is None:
        return False
    if not isinstance(saved_payload, list):
        return False
    current_fingerprints = build_file_fingerprints(current_paths)
    return saved_payload == current_fingerprints


def evaluate_visible_structure_goals(
    curves: List[DVHCurve],
    goals_by_structure: Dict[str, List[StructureGoal]],
    selected_names: List[str],
) -> Dict[str, List[StructureGoalEvaluation]]:
    selected_name_set = {normalize_structure_name(name) for name in selected_names}
    visible_curves = [
        curve for curve in curves if normalize_structure_name(curve.name) in selected_name_set
    ]
    return evaluate_structure_goals(visible_curves, goals_by_structure)


class DVHComputationSignals(QtCore.QObject):
    finished = QtCore.Signal(int, object, object, float)
    failed = QtCore.Signal(int, str, float)


class DVHComputationTask(QtCore.QRunnable):
    def __init__(
        self,
        request_id: int,
        ct: CTVolume,
        dose: DoseVolume,
        rtstruct: RTStructData,
        dose_ct_volume: Optional[np.ndarray],
        mask_cache: Optional[List[Dict[int, np.ndarray]]],
        dvh_mode: str,
    ):
        super().__init__()
        self.setAutoDelete(False)
        self.request_id = request_id
        self.ct = ct
        self.dose = dose
        self.rtstruct = rtstruct
        self.dose_ct_volume = dose_ct_volume
        self.mask_cache = mask_cache
        self.dvh_mode = dvh_mode
        self.signals = DVHComputationSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.dose_ct_volume = None
        self.mask_cache = None

    def _release_references(self) -> None:
        self.dose_ct_volume = None
        self.mask_cache = None
        self.ct = None  # type: ignore[assignment]
        self.dose = None  # type: ignore[assignment]
        self.rtstruct = None  # type: ignore[assignment]

    def run(self):
        start = perf_counter()
        try:
            if self._cancelled:
                return
            mask_cache = self.mask_cache
            if mask_cache is None or len(mask_cache) != len(self.rtstruct.structures):
                mask_cache = build_structure_mask_cache(self.rtstruct, self.ct.rows, self.ct.cols)
            if self._cancelled:
                return
            curves = compute_dvh_curves(
                self.ct,
                self.dose,
                self.rtstruct,
                dose_ct_volume=self.dose_ct_volume,
                mask_cache=mask_cache,
                mode=self.dvh_mode,
            )
        except Exception as exc:
            if self._cancelled:
                self._release_references()
                return
            self.signals.failed.emit(self.request_id, str(exc), perf_counter() - start)
            self._release_references()
            return
        finally:
            if self._cancelled:
                self._release_references()

        if self._cancelled:
            return
        self.signals.finished.emit(self.request_id, curves, mask_cache, perf_counter() - start)
        self._release_references()


class DVHComputationManager(QtCore.QObject):
    finished = QtCore.Signal(int, object, object, float)
    failed = QtCore.Signal(int, str, float)

    def __init__(self, thread_pool: Optional[QtCore.QThreadPool] = None):
        super().__init__()
        self.thread_pool = thread_pool or QtCore.QThreadPool.globalInstance()
        self.request_id = 0
        self.active_jobs: Dict[int, DVHComputationTask] = {}
        self.pending_request: Optional[
            Tuple[
                int,
                CTVolume,
                DoseVolume,
                RTStructData,
                Optional[np.ndarray],
                Optional[List[Dict[int, np.ndarray]]],
                str,
            ]
        ] = None

    def invalidate(self) -> None:
        self.request_id += 1
        self.pending_request = None

    def cancel_all(self) -> None:
        self.request_id += 1
        self.pending_request = None
        for task in self.active_jobs.values():
            task.cancel()
        self.active_jobs.clear()

    def start(
        self,
        ct: CTVolume,
        dose: DoseVolume,
        rtstruct: RTStructData,
        dose_ct_volume: Optional[np.ndarray],
        mask_cache: Optional[List[Dict[int, np.ndarray]]],
        dvh_mode: str,
    ) -> int:
        self.request_id += 1
        request_id = self.request_id
        request = (
            request_id,
            ct,
            dose,
            rtstruct,
            dose_ct_volume,
            mask_cache,
            dvh_mode,
        )
        if self.active_jobs:
            self.pending_request = request
            return request_id
        self._start_request(*request)
        return request_id

    def _start_request(
        self,
        request_id: int,
        ct: CTVolume,
        dose: DoseVolume,
        rtstruct: RTStructData,
        dose_ct_volume: Optional[np.ndarray],
        mask_cache: Optional[List[Dict[int, np.ndarray]]],
        dvh_mode: str,
    ) -> None:
        task = DVHComputationTask(
            request_id,
            ct,
            dose,
            rtstruct,
            dose_ct_volume,
            mask_cache,
            dvh_mode,
        )
        task.signals.finished.connect(self._on_task_finished)
        task.signals.failed.connect(self._on_task_failed)
        self.active_jobs[request_id] = task
        self.thread_pool.start(task)

    def is_current(self, request_id: int) -> bool:
        return request_id == self.request_id

    def _launch_pending_if_idle(self) -> None:
        if self.active_jobs or self.pending_request is None:
            return
        request = self.pending_request
        self.pending_request = None
        self._start_request(*request)

    def _on_task_finished(
        self,
        request_id: int,
        curves: object,
        mask_cache: object,
        duration_s: float,
    ) -> None:
        self.active_jobs.pop(request_id, None)
        self._launch_pending_if_idle()
        self.finished.emit(request_id, curves, mask_cache, duration_s)

    def _on_task_failed(self, request_id: int, error_message: str, duration_s: float) -> None:
        self.active_jobs.pop(request_id, None)
        self._launch_pending_if_idle()
        self.failed.emit(request_id, error_message, duration_s)


class StructureListItemWidget(QtWidgets.QWidget):
    checkedChanged = QtCore.Signal(bool)

    def __init__(
        self,
        name: str,
        color_rgb: Tuple[int, int, int],
        checked: bool,
        goal_lines: List[Tuple[str, Optional[str]]],
        show_checkbox: bool = True,
        name_font_point_size: Optional[int] = None,
        goal_font_point_size: Optional[int] = None,
        leading_button_text: Optional[str] = None,
        leading_button_callback: Optional[Callable[[], None]] = None,
        trailing_button_text: Optional[str] = None,
        trailing_button_callback: Optional[Callable[[], None]] = None,
        inline_goals: bool = False,
        inline_goals_compact: bool = False,
        secondary_text: Optional[str] = None,
        secondary_text_color: Optional[str] = None,
    ):
        super().__init__()
        self.show_checkbox = show_checkbox
        self.inline_goals = inline_goals
        self.inline_goals_compact = inline_goals_compact
        self.checkbox: Optional[QtWidgets.QCheckBox] = None
        self.title_label: Optional[QtWidgets.QLabel] = None
        self.inline_goal_label: Optional[QtWidgets.QLabel] = None
        self.secondary_label: Optional[QtWidgets.QLabel] = None

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(6, 4, 6, 4)
        self.layout.setSpacing(4)

        if show_checkbox:
            title_row = QtWidgets.QWidget()
            title_row_layout = QtWidgets.QHBoxLayout(title_row)
            title_row_layout.setContentsMargins(0, 0, 0, 0)
            title_row_layout.setSpacing(6)
            self.checkbox = QtWidgets.QCheckBox(name)
            self.checkbox.setChecked(checked)
            self.checkbox.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self.checkbox.setStyleSheet(
                f"QCheckBox {{ color: rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}); font-weight: 600; padding: 2px 0px; }}"
            )
            if name_font_point_size is not None:
                font = self.checkbox.font()
                font.setPointSize(name_font_point_size)
                self.checkbox.setFont(font)
            self.checkbox.stateChanged.connect(self._on_checkbox_state_changed)
            title_row_layout.addWidget(self.checkbox, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            self.secondary_label = QtWidgets.QLabel("")
            self.secondary_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.secondary_label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            if goal_font_point_size is not None:
                font = self.secondary_label.font()
                font.setPointSize(goal_font_point_size)
                self.secondary_label.setFont(font)
            title_row_layout.addSpacing(max(8, self.checkbox.fontMetrics().horizontalAdvance("  ")))
            title_row_layout.addWidget(self.secondary_label, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            title_row_layout.addStretch(1)
            self.layout.addWidget(title_row, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        else:
            title_row = QtWidgets.QWidget()
            title_row_layout = QtWidgets.QHBoxLayout(title_row)
            title_row_layout.setContentsMargins(0, 0, 0, 0)
            title_row_layout.setSpacing(6)
            if leading_button_text:
                button = QtWidgets.QPushButton(leading_button_text)
                button.setFixedWidth(34)
                button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
                if leading_button_callback is not None:
                    button.clicked.connect(leading_button_callback)
                title_row_layout.addWidget(button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            self.title_label = QtWidgets.QLabel(name)
            self.title_label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            title_style = (
                f"color: rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]});"
                " font-weight: 600; padding: 2px 0px;"
            )
            self.title_label.setStyleSheet(title_style)
            if name_font_point_size is not None:
                font = self.title_label.font()
                font.setPointSize(name_font_point_size)
                self.title_label.setFont(font)
            title_row_layout.addWidget(self.title_label, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            if inline_goals:
                self.inline_goal_label = QtWidgets.QLabel("")
                self.inline_goal_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.inline_goal_label.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Fixed,
                    QtWidgets.QSizePolicy.Policy.Fixed,
                )
                if goal_font_point_size is not None:
                    font = self.inline_goal_label.font()
                    font.setPointSize(goal_font_point_size)
                    self.inline_goal_label.setFont(font)
                if self.inline_goals_compact:
                    spacing_width = max(8, self.title_label.fontMetrics().horizontalAdvance("  "))
                    title_row_layout.addSpacing(spacing_width)
                    title_row_layout.addWidget(self.inline_goal_label, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
                else:
                    title_row_layout.addStretch(1)
                    title_row_layout.addWidget(self.inline_goal_label, 0, QtCore.Qt.AlignmentFlag.AlignRight)
            title_row_layout.addStretch(1)
            if trailing_button_text:
                button = QtWidgets.QPushButton(trailing_button_text)
                button.setFixedWidth(34)
                button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
                if trailing_button_callback is not None:
                    button.clicked.connect(trailing_button_callback)
                title_row_layout.addWidget(button, 0, QtCore.Qt.AlignmentFlag.AlignRight)
            self.layout.addWidget(title_row, 0, QtCore.Qt.AlignmentFlag.AlignTop)

        self.goals_widget = QtWidgets.QWidget()
        self.goals_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.goals_layout = QtWidgets.QVBoxLayout(self.goals_widget)
        left_margin = 28 if show_checkbox else 10
        self.goals_layout.setContentsMargins(left_margin, 0, 6, 2)
        self.goals_layout.setSpacing(3)
        self.layout.addWidget(self.goals_widget, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self.goal_font_point_size = goal_font_point_size

        self.set_goal_lines(goal_lines)
        self.set_secondary_text(secondary_text, secondary_text_color)

    def sizeHint(self) -> QtCore.QSize:
        self.layout.activate()
        self.goals_layout.activate()
        hint = self.layout.sizeHint()
        return QtCore.QSize(hint.width(), hint.height() + 4)

    def is_checked(self) -> bool:
        if self.checkbox is None:
            return True
        return self.checkbox.isChecked()

    def set_checked(self, checked: bool) -> None:
        if self.checkbox is None:
            return
        blocker = QtCore.QSignalBlocker(self.checkbox)
        self.checkbox.setChecked(checked)
        del blocker

    def set_goal_lines(self, goal_lines: List[Tuple[str, Optional[str]]]) -> None:
        if self.inline_goals and self.inline_goal_label is not None:
            inline_text = goal_lines[0][0] if goal_lines else ""
            self.inline_goal_label.setText(inline_text)
            if goal_lines and goal_lines[0][1] is not None:
                self.inline_goal_label.setStyleSheet(f"color: {goal_lines[0][1]};")
            else:
                self.inline_goal_label.setStyleSheet("")
            self.goals_widget.setVisible(False)
            self.updateGeometry()
            return

        while self.goals_layout.count():
            item = self.goals_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.goals_widget.setVisible(bool(goal_lines))
        for text, color_name in goal_lines:
            label = QtWidgets.QLabel(text)
            label.setWordWrap(False)
            label.setContentsMargins(0, 0, 0, 1)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
            label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            if self.goal_font_point_size is not None:
                font = label.font()
                font.setPointSize(self.goal_font_point_size)
                label.setFont(font)
            if color_name is not None:
                label.setStyleSheet(f"color: {color_name};")
            self.goals_layout.addWidget(label, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self.updateGeometry()

    def set_secondary_text(self, text: Optional[str], color_name: Optional[str] = None) -> None:
        if self.secondary_label is None:
            return
        display_text = (text or "").strip()
        self.secondary_label.setText(display_text)
        self.secondary_label.setVisible(bool(display_text))
        if color_name:
            self.secondary_label.setStyleSheet(f"color: {color_name};")
        else:
            self.secondary_label.setStyleSheet("")
        self.updateGeometry()

    def _on_checkbox_state_changed(self, _state: int) -> None:
        if self.checkbox is not None:
            self.checkedChanged.emit(self.checkbox.isChecked())

    def wheelEvent(self, event) -> None:
        scroll_area = self._find_parent_scroll_area()
        if scroll_area is None:
            super().wheelEvent(event)
            return

        scroll_bar = scroll_area.verticalScrollBar()
        pixel_delta = event.pixelDelta().y()
        angle_delta = event.angleDelta().y()
        if pixel_delta:
            scroll_bar.setValue(scroll_bar.value() - pixel_delta)
            event.accept()
            return
        if angle_delta:
            single_step = max(scroll_bar.singleStep(), 20)
            steps = angle_delta / 120.0
            scroll_bar.setValue(scroll_bar.value() - int(round(steps * single_step)))
            event.accept()
            return

        super().wheelEvent(event)

    def _find_parent_scroll_area(self) -> Optional[QtWidgets.QAbstractScrollArea]:
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QtWidgets.QAbstractScrollArea):
                return parent
            parent = parent.parentWidget()
        return None


class StructureListManager(QtCore.QObject):
    visibilityChanged = QtCore.Signal()

    def __init__(
        self,
        collections: List[Tuple[str, QtWidgets.QListWidget]],
        *,
        interactive: bool = True,
        name_font_point_size: Optional[int] = None,
        goal_font_point_size: Optional[int] = None,
    ):
        super().__init__()
        self._collections = collections
        self._interactive = interactive
        self._name_font_point_size = name_font_point_size
        self._goal_font_point_size = goal_font_point_size
        self._widget_maps: Dict[str, Dict[str, StructureListItemWidget]] = {
            tag: {} for tag, _ in collections
        }
        self._item_maps: Dict[str, Dict[str, QtWidgets.QListWidgetItem]] = {
            tag: {} for tag, _ in collections
        }

    def set_structures(
        self,
        rtstruct: Optional[RTStructData],
        goal_line_getter: Callable[[str], List[Tuple[str, Optional[str]]]],
        default_visibility_resolver: Optional[Callable[[str], bool]] = None,
        show_checkbox_resolver: Optional[Callable[[str], bool]] = None,
        item_options_getter: Optional[Callable[[str], Dict[str, object]]] = None,
    ) -> None:
        previous_visibility = self.visibility_map()
        for tag, list_widget in self._collections:
            self._populate_list(
                list_widget,
                self._widget_maps[tag],
                self._item_maps[tag],
                previous_visibility,
                tag,
                rtstruct,
                goal_line_getter,
                default_visibility_resolver,
                show_checkbox_resolver,
                item_options_getter,
            )
        self.schedule_layout_refresh()

    def update_goal_lines(
        self,
        rtstruct: Optional[RTStructData],
        goal_line_getter: Callable[[str], List[Tuple[str, Optional[str]]]],
    ) -> None:
        if rtstruct is None:
            return

        for tag, _list_widget in self._collections:
            widget_map = self._widget_maps[tag]
            item_map = self._item_maps[tag]
            for structure in rtstruct.structures:
                normalized_name = normalize_structure_name(structure.name)
                widget = widget_map.get(normalized_name)
                item = item_map.get(normalized_name)
                if widget is None or item is None:
                    continue
                widget.set_goal_lines(goal_line_getter(normalized_name))
                item.setSizeHint(widget.sizeHint())
        self.schedule_layout_refresh()

    def update_secondary_texts(
        self,
        rtstruct: Optional[RTStructData],
        secondary_text_getter: Callable[[str], Tuple[Optional[str], Optional[str]]],
    ) -> None:
        if rtstruct is None:
            return

        for tag, _list_widget in self._collections:
            widget_map = self._widget_maps[tag]
            item_map = self._item_maps[tag]
            for structure in rtstruct.structures:
                normalized_name = normalize_structure_name(structure.name)
                widget = widget_map.get(normalized_name)
                item = item_map.get(normalized_name)
                if widget is None or item is None:
                    continue
                text, color_name = secondary_text_getter(normalized_name)
                widget.set_secondary_text(text, color_name)
                item.setSizeHint(widget.sizeHint())
        self.schedule_layout_refresh()

    def visibility_map(self) -> Dict[str, bool]:
        visibility: Dict[str, bool] = {}
        for tag, _list_widget in self._collections:
            for normalized_name, widget in self._widget_maps[tag].items():
                visibility[normalized_name] = widget.is_checked()
        return visibility

    def is_visible(self, normalized_name: str) -> bool:
        for tag, _list_widget in self._collections:
            widget = self._widget_maps[tag].get(normalized_name)
            if widget is not None:
                return widget.is_checked()
        return True

    def set_checked(self, normalized_name: str, checked: bool, *, emit_signal: bool = False) -> bool:
        changed = False
        for tag, _list_widget in self._collections:
            widget = self._widget_maps[tag].get(normalized_name)
            if widget is None:
                continue
            if widget.is_checked() == checked:
                continue
            widget.set_checked(checked)
            changed = True
        if changed and emit_signal:
            self.visibilityChanged.emit()
        return changed

    def set_checked_names(self, checked_names: List[str], *, emit_signal: bool = False) -> bool:
        changed = False
        checked_name_set = {normalize_structure_name(name) for name in checked_names}
        seen_names = set()
        for tag, _list_widget in self._collections:
            for normalized_name, widget in self._widget_maps[tag].items():
                if normalized_name in seen_names:
                    continue
                seen_names.add(normalized_name)
                desired_checked = normalized_name in checked_name_set
                if widget.is_checked() == desired_checked:
                    continue
                self.set_checked(normalized_name, desired_checked, emit_signal=False)
                changed = True
        if changed and emit_signal:
            self.visibilityChanged.emit()
        return changed

    def set_enabled(self, enabled: bool) -> None:
        for _tag, list_widget in self._collections:
            list_widget.setEnabled(enabled)

    def refresh_layout(self, *_args) -> None:
        for tag, list_widget in self._collections:
            widget_map = self._widget_maps[tag]
            item_map = self._item_maps[tag]
            for normalized_name, widget in widget_map.items():
                item = item_map.get(normalized_name)
                if item is None:
                    continue
                widget.ensurePolished()
                widget.adjustSize()
                item.setSizeHint(widget.sizeHint())
            list_widget.doItemsLayout()
            list_widget.viewport().update()

    def schedule_layout_refresh(self) -> None:
        self.refresh_layout()
        QtCore.QTimer.singleShot(0, self.refresh_layout)

    def _populate_list(
        self,
        list_widget: QtWidgets.QListWidget,
        widget_map: Dict[str, StructureListItemWidget],
        item_map: Dict[str, QtWidgets.QListWidgetItem],
        previous_visibility: Dict[str, bool],
        source_tag: str,
        rtstruct: Optional[RTStructData],
        goal_line_getter: Callable[[str], List[Tuple[str, Optional[str]]]],
        default_visibility_resolver: Optional[Callable[[str], bool]],
        show_checkbox_resolver: Optional[Callable[[str], bool]],
        item_options_getter: Optional[Callable[[str], Dict[str, object]]],
    ) -> None:
        list_widget.clear()
        widget_map.clear()
        item_map.clear()

        if rtstruct is None:
            return

        for structure in rtstruct.structures:
            normalized_name = normalize_structure_name(structure.name)
            fallback_visible = (
                default_visibility_resolver(normalized_name)
                if default_visibility_resolver is not None
                else normalized_name.startswith("PTV")
            )
            default_visible = previous_visibility.get(normalized_name, fallback_visible)
            show_checkbox = self._interactive and (
                show_checkbox_resolver(normalized_name)
                if show_checkbox_resolver is not None
                else True
            )
            item_options = item_options_getter(normalized_name) if item_options_getter is not None else {}
            item = QtWidgets.QListWidgetItem()
            widget = StructureListItemWidget(
                structure.name,
                structure.color_rgb,
                default_visible,
                goal_line_getter(normalized_name),
                show_checkbox=show_checkbox,
                name_font_point_size=self._name_font_point_size,
                goal_font_point_size=self._goal_font_point_size,
                **item_options,
            )
            if show_checkbox:
                widget.checkedChanged.connect(
                    lambda checked, name=normalized_name, source=source_tag: self._on_widget_checked_changed(
                        name,
                        source,
                        checked,
                    )
                )
            list_widget.addItem(item)
            list_widget.setItemWidget(item, widget)
            item.setSizeHint(widget.sizeHint())
            widget_map[normalized_name] = widget
            item_map[normalized_name] = item

    def _on_widget_checked_changed(self, normalized_name: str, source_tag: str, checked: bool) -> None:
        for tag, _list_widget in self._collections:
            if tag == source_tag:
                continue
            widget = self._widget_maps[tag].get(normalized_name)
            if widget is not None and widget.is_checked() != checked:
                widget.set_checked(checked)
        self.visibilityChanged.emit()
