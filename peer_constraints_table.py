from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from peer_helpers import (
    evaluate_structure_goal,
    normalize_structure_name,
    parse_goal_value,
    parse_goal_value_range,
    parse_v_metric_threshold_gy,
    volume_cc_at_dose_gy,
)
from peer_models import DVHCurve, RTStructData, StructureGoal, StructureGoalEvaluation


@dataclass(frozen=True)
class ConstraintTablePresentationRow:
    background_color: QtGui.QColor
    oar: str
    metric: str
    goal_text: str
    actual_text: str
    evaluation_status: str
    note_key: str
    note_title: str
    note_text: str
    is_custom_only: bool


@dataclass(frozen=True)
class ConstraintEditorPreviewState:
    text: str
    color_name: str
    add_enabled: bool


def get_constraints_table_column_widths(viewport_width: int) -> list[int]:
    viewport_width = max(int(viewport_width), 1)
    column_widths = [
        max(1, int(round(viewport_width * 0.20))),
        max(1, int(round(viewport_width * 0.05))),
        max(1, int(round(viewport_width * 0.09))),
        max(1, int(round(viewport_width * 0.06))),
        max(1, int(round(viewport_width * 0.54))),
        max(72, int(round(viewport_width * 0.06))),
    ]
    width_delta = viewport_width - sum(column_widths)
    column_widths[4] += width_delta
    return column_widths


def build_initial_constraint_editor_state(
    structure_names: Sequence[str],
) -> Optional[dict[str, str]]:
    if not structure_names:
        return None
    return {
        "structure_name": str(structure_names[0]),
        "metric": "",
        "goal_text": "",
    }


def parse_constraint_goal_input(goal_text: str) -> Optional[Tuple[str, str]]:
    match = re.match(r"^\s*(<=|>=|<|>|==|=)\s*(.+?)\s*$", goal_text)
    if match is None:
        return None
    comparator = match.group(1).strip()
    value_text = match.group(2).strip()
    if not value_text:
        return None
    return comparator, value_text


def get_constraint_goal_key(goal: StructureGoal) -> Tuple[str, str, str]:
    return (
        goal.metric.strip().upper().replace(" ", ""),
        goal.comparator.strip(),
        goal.value_text.strip().upper(),
    )


def build_custom_constraint_from_editor(
    constraint_editor_state: Optional[Mapping[str, str]],
) -> Optional[Tuple[str, StructureGoal]]:
    if constraint_editor_state is None:
        return None

    structure_name = constraint_editor_state.get("structure_name", "").strip()
    metric = constraint_editor_state.get("metric", "").strip()
    goal_input = constraint_editor_state.get("goal_text", "").strip()
    if not structure_name or not metric:
        return None

    parsed_goal = parse_constraint_goal_input(goal_input)
    if parsed_goal is None:
        return None
    comparator, value_text = parsed_goal
    normalized_name = normalize_structure_name(structure_name)
    return normalized_name, StructureGoal(
        structure_name=structure_name,
        metric=metric,
        comparator=comparator,
        value_text=value_text,
    )


def custom_constraint_exists(
    normalized_name: str,
    goal: StructureGoal,
    *,
    structure_goals_by_name: Mapping[str, Sequence[StructureGoal]],
) -> bool:
    goal_key = get_constraint_goal_key(goal)
    for existing_goal in structure_goals_by_name.get(normalized_name, []):
        existing_key = get_constraint_goal_key(existing_goal)
        if existing_key == goal_key:
            return True
    return False


def build_constraint_editor_preview_state(
    constraint_editor_state: Optional[Mapping[str, str]],
    *,
    structure_goals_by_name: Mapping[str, Sequence[StructureGoal]],
    get_curve_for_name: Callable[[str], Optional[DVHCurve]],
    dvh_structure_is_visible: Callable[[str], bool],
    structure_goal_line_color: Callable[[StructureGoalEvaluation], Optional[str]],
) -> ConstraintEditorPreviewState:
    if constraint_editor_state is None:
        return ConstraintEditorPreviewState("", "#bdbdbd", False)

    structure_name = constraint_editor_state.get("structure_name", "").strip()
    metric_text = constraint_editor_state.get("metric", "").strip()
    goal_input = constraint_editor_state.get("goal_text", "").strip()

    if not structure_name or not metric_text or not goal_input:
        return ConstraintEditorPreviewState("Enter metric and goal.", "#bdbdbd", False)

    built_goal = build_custom_constraint_from_editor(constraint_editor_state)
    if built_goal is None:
        return ConstraintEditorPreviewState("Use goal like <= 30 Gy", "#ffb86b", False)

    normalized_name, preview_goal = built_goal
    if custom_constraint_exists(
        normalized_name,
        preview_goal,
        structure_goals_by_name=structure_goals_by_name,
    ):
        return ConstraintEditorPreviewState("Constraint already exists.", "#ffb86b", False)

    curve = get_curve_for_name(normalized_name)
    if curve is None:
        if dvh_structure_is_visible(normalized_name):
            return ConstraintEditorPreviewState("Will calculate when DVH finishes.", "#bdbdbd", True)
        return ConstraintEditorPreviewState("Will calculate after adding.", "#bdbdbd", True)

    evaluation = evaluate_structure_goal(curve, preview_goal)
    color_name = structure_goal_line_color(evaluation) or "#d9d9d9"
    return ConstraintEditorPreviewState(evaluation.actual_text, color_name, True)


def get_constraint_evaluations_for_structure(
    normalized_name: str,
    goals: Sequence[StructureGoal],
    *,
    structure_goal_evaluations: Mapping[str, Sequence[StructureGoalEvaluation]],
    dvh_structure_goal_evaluation_cache: dict[str, list[StructureGoalEvaluation]],
    get_curve_for_name: Callable[[str], Optional[DVHCurve]],
) -> list[StructureGoalEvaluation]:
    evaluations = list(structure_goal_evaluations.get(normalized_name, []))
    if evaluations and len(evaluations) >= len(goals):
        return evaluations

    curve = get_curve_for_name(normalized_name)
    if curve is not None and goals:
        computed = [evaluate_structure_goal(curve, goal) for goal in goals]
        dvh_structure_goal_evaluation_cache[normalized_name] = list(computed)
        return computed

    cached = dvh_structure_goal_evaluation_cache.get(normalized_name, [])
    if cached:
        return cached

    return evaluations


def compose_constraint_note_text(computed_note_text: str, stored_note_text: str) -> str:
    computed = computed_note_text.strip()
    stored = stored_note_text.strip()
    if computed and stored:
        if stored == computed or stored.startswith(f"{computed}    "):
            return stored
        return f"{computed}    {stored}"
    if computed:
        return computed
    return stored


def prostate_constraint_summary_enabled(constraints_sheet_name: str) -> bool:
    return "PROSTATE" in normalize_structure_name(constraints_sheet_name or "")


def get_min_bladder_volume_note_text(
    *,
    constraints_sheet_name: str,
    structure_goals_by_name: Mapping[str, Sequence[StructureGoal]],
    get_curve_for_name: Callable[[str], Optional[DVHCurve]],
) -> str:
    if not prostate_constraint_summary_enabled(constraints_sheet_name):
        return ""

    normalized_name = "BLADDER"
    curve = get_curve_for_name(normalized_name)
    if curve is None or curve.volume_cc <= 0.0:
        return ""

    goals = structure_goals_by_name.get(normalized_name, [])
    required_volume_cc = 0.0
    found_percent_volume_constraint = False
    for goal in goals:
        metric_key = goal.metric.strip().upper().replace(" ", "")
        dose_threshold_gy = parse_v_metric_threshold_gy(metric_key)
        if dose_threshold_gy is None:
            continue
        comparator = goal.comparator.strip()
        if comparator not in {"<", "<=", "=", "=="}:
            continue
        goal_value, goal_unit = parse_goal_value(goal.value_text)
        if goal_value is None:
            continue
        actual_volume_cc = float(volume_cc_at_dose_gy(curve, dose_threshold_gy))
        if goal_unit == "%":
            goal_fraction = goal_value / 100.0
            if goal_fraction <= 0.0:
                continue
            required_volume_cc = max(required_volume_cc, actual_volume_cc / goal_fraction)
            found_percent_volume_constraint = True
        elif goal_unit == "CC":
            if actual_volume_cc > goal_value + 1e-6:
                return "Min volume n/a"

    if not found_percent_volume_constraint or required_volume_cc <= 0.0:
        return ""
    return f"Min volume {int(math.ceil(required_volume_cc - 1e-9))} cc"


def get_computed_constraint_note_text(
    normalized_name: str,
    goals: Sequence[StructureGoal],
    goal_index: int,
    *,
    evaluation: Optional[StructureGoalEvaluation],
    constraints_sheet_name: str,
    structure_goals_by_name: Mapping[str, Sequence[StructureGoal]],
    get_curve_for_name: Callable[[str], Optional[DVHCurve]],
) -> str:
    notes: list[str] = []
    if 0 <= goal_index < len(goals):
        goal = goals[goal_index]
        variation_start, variation_end, _variation_unit = parse_goal_value_range(goal.value_text)
        if (
            variation_start is not None
            and variation_end is not None
            and evaluation is not None
            and evaluation.status == "variation"
        ):
            notes.append("Acceptable Variation")
    if normalized_name == "BLADDER" and goal_index == 0 and goals:
        bladder_note = get_min_bladder_volume_note_text(
            constraints_sheet_name=constraints_sheet_name,
            structure_goals_by_name=structure_goals_by_name,
            get_curve_for_name=get_curve_for_name,
        )
        if bladder_note:
            notes.append(bladder_note)
    return "    ".join(notes)


def build_constraints_table_presentation_rows(
    *,
    rtstruct: Optional[RTStructData],
    structure_goals_by_name: Mapping[str, Sequence[StructureGoal]],
    structure_goal_evaluations: Mapping[str, Sequence[StructureGoalEvaluation]],
    dvh_structure_goal_evaluation_cache: dict[str, list[StructureGoalEvaluation]],
    constraint_notes: Mapping[str, str],
    constraints_sheet_name: str,
    get_curve_for_name: Callable[[str], Optional[DVHCurve]],
    get_constraint_note_key: Callable[[str, StructureGoal], str],
    is_custom_only_constraint: Callable[[str, StructureGoal], bool],
) -> list[ConstraintTablePresentationRow]:
    if rtstruct is None:
        return []

    rows: list[ConstraintTablePresentationRow] = []
    row_backgrounds = [QtGui.QColor(8, 8, 8), QtGui.QColor(34, 34, 34)]
    structure_group_index = 0
    for structure in rtstruct.structures:
        normalized_name = normalize_structure_name(structure.name)
        goals = list(structure_goals_by_name.get(normalized_name, []))
        if not goals:
            continue
        evaluations = get_constraint_evaluations_for_structure(
            normalized_name,
            goals,
            structure_goal_evaluations=structure_goal_evaluations,
            dvh_structure_goal_evaluation_cache=dvh_structure_goal_evaluation_cache,
            get_curve_for_name=get_curve_for_name,
        )
        background_color = row_backgrounds[structure_group_index % len(row_backgrounds)]
        for goal_index, goal in enumerate(goals):
            evaluation = evaluations[goal_index] if goal_index < len(evaluations) else None
            note_key = get_constraint_note_key(normalized_name, goal)
            note_title = f"{structure.name} | {goal.metric} {goal.comparator} {goal.value_text}"
            computed_note_text = get_computed_constraint_note_text(
                normalized_name,
                goals,
                goal_index,
                evaluation=evaluation,
                constraints_sheet_name=constraints_sheet_name,
                structure_goals_by_name=structure_goals_by_name,
                get_curve_for_name=get_curve_for_name,
            )
            note_text = compose_constraint_note_text(
                computed_note_text,
                constraint_notes.get(note_key, ""),
            )
            rows.append(
                ConstraintTablePresentationRow(
                    background_color=background_color,
                    oar=structure.name if goal_index == 0 else "",
                    metric=goal.metric,
                    goal_text=f"{goal.comparator.strip()} {goal.value_text.strip()}".strip(),
                    actual_text=evaluation.actual_text if evaluation is not None else "",
                    evaluation_status=evaluation.status if evaluation is not None else "",
                    note_key=note_key,
                    note_title=note_title,
                    note_text=note_text,
                    is_custom_only=is_custom_only_constraint(normalized_name, goal),
                )
            )
        structure_group_index += 1
    return rows


def create_constraint_note_button_widget(
    note_key: str,
    title: str,
    note_text: str,
    background_color: QtGui.QColor,
    *,
    on_edit_note: Callable[[str, str], None],
) -> QtWidgets.QWidget:
    button = QtWidgets.QPushButton("Note")
    button.setFixedWidth(56)
    cleaned_note = note_text.strip()
    if cleaned_note:
        font = button.font()
        font.setBold(True)
        button.setFont(font)
        button.setToolTip(cleaned_note)
    else:
        button.setToolTip(f"Add note for {title}")
    button.setStyleSheet(
        "QPushButton {"
        f" background-color: {background_color.lighter(112).name()};"
        f" border: 1px solid {background_color.lighter(150).name()};"
        " padding: 2px 6px;"
        "}"
    )
    button.clicked.connect(
        lambda _checked=False, key=note_key, dialog_title=title: on_edit_note(key, dialog_title)
    )
    container = QtWidgets.QWidget()
    container.setAutoFillBackground(True)
    palette = container.palette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, background_color)
    container.setPalette(palette)
    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(2, 0, 2, 0)
    layout.addStretch(1)
    layout.addWidget(button)
    layout.addStretch(1)
    return container


def create_constraint_text_item(
    text: str,
    background_color: QtGui.QColor,
    *,
    column_index: int,
    evaluation_status: str,
    is_custom_only: bool,
    tooltip_text: Optional[str] = None,
) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
    item.setBackground(background_color)
    item.setForeground(QtGui.QColor("#f2f2f2"))
    if column_index == 0 and text:
        item.setForeground(QtGui.QColor(255, 255, 255))
        font = item.font()
        font.setBold(True)
        item.setFont(font)
    elif is_custom_only and column_index in {1, 2}:
        item.setForeground(QtGui.QColor("#ffd54a"))
    elif column_index == 3:
        if evaluation_status == "pass":
            item.setForeground(QtGui.QColor("#63c174"))
            item.setBackground(QtGui.QColor("#18351f"))
        elif evaluation_status == "variation":
            item.setForeground(QtGui.QColor("#ffd54a"))
            item.setBackground(QtGui.QColor("#4a3f10"))
        elif evaluation_status == "fail":
            item.setForeground(QtGui.QColor("#ff6b6b"))
            item.setBackground(QtGui.QColor("#3e1616"))
    if tooltip_text:
        item.setToolTip(tooltip_text)
    return item


def populate_constraints_table_rows(
    table: QtWidgets.QTableWidget,
    rows: Sequence[ConstraintTablePresentationRow],
    *,
    row_offset: int,
    on_edit_note: Callable[[str, str], None],
) -> None:
    table.setRowCount(len(rows) + row_offset)
    for row_index, row in enumerate(rows, start=row_offset):
        values = [row.oar, row.metric, row.goal_text, row.actual_text]
        for column_index, text in enumerate(values):
            table.setItem(
                row_index,
                column_index,
                create_constraint_text_item(
                    text,
                    row.background_color,
                    column_index=column_index,
                    evaluation_status=row.evaluation_status,
                    is_custom_only=row.is_custom_only,
                ),
            )
        table.setItem(
            row_index,
            4,
            create_constraint_text_item(
                row.note_text,
                row.background_color,
                column_index=4,
                evaluation_status=row.evaluation_status,
                is_custom_only=row.is_custom_only,
                tooltip_text=row.note_text if row.note_text else None,
            ),
        )
        table.setCellWidget(
            row_index,
            5,
            create_constraint_note_button_widget(
                row.note_key,
                row.note_title,
                row.note_text,
                row.background_color,
                on_edit_note=on_edit_note,
            ),
        )


def refresh_constraints_table(
    table: QtWidgets.QTableWidget,
    rows: Sequence[ConstraintTablePresentationRow],
    *,
    constraint_editor_state: Optional[Mapping[str, str]],
    structure_names: Sequence[str],
    on_edit_note: Callable[[str, str], None],
    on_field_change: Callable[[str, str], None],
    on_commit: Callable[[], None],
    on_cancel: Callable[[], None],
) -> dict[str, QtWidgets.QWidget]:
    table.setRowCount(0)
    row_offset = 1 if constraint_editor_state is not None else 0
    populate_constraints_table_rows(
        table,
        rows,
        row_offset=row_offset,
        on_edit_note=on_edit_note,
    )
    if constraint_editor_state is not None:
        return populate_constraint_editor_row(
            table,
            0,
            constraint_editor_state=constraint_editor_state,
            structure_names=structure_names,
            on_field_change=on_field_change,
            on_commit=on_commit,
            on_cancel=on_cancel,
            background_color=QtGui.QColor(20, 20, 20),
        )
    return {}


def create_constraint_editor_action_widget(
    background_color: QtGui.QColor,
    *,
    on_commit: Callable[[], None],
    on_cancel: Callable[[], None],
) -> tuple[QtWidgets.QWidget, QtWidgets.QPushButton]:
    add_button = QtWidgets.QPushButton("Add")
    add_button.setFixedWidth(52)
    add_button.clicked.connect(on_commit)

    cancel_button = QtWidgets.QPushButton("X")
    cancel_button.setFixedWidth(28)
    cancel_button.clicked.connect(on_cancel)

    button_style = (
        "QPushButton {"
        f" background-color: {background_color.lighter(112).name()};"
        f" border: 1px solid {background_color.lighter(150).name()};"
        " padding: 2px 6px;"
        "}"
    )
    add_button.setStyleSheet(button_style)
    cancel_button.setStyleSheet(button_style)

    container = QtWidgets.QWidget()
    container.setAutoFillBackground(True)
    palette = container.palette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, background_color)
    container.setPalette(palette)

    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(2, 0, 2, 0)
    layout.setSpacing(4)
    layout.addStretch(1)
    layout.addWidget(add_button)
    layout.addWidget(cancel_button)
    layout.addStretch(1)
    return container, add_button


def populate_constraint_editor_row(
    table: QtWidgets.QTableWidget,
    row_index: int,
    *,
    constraint_editor_state: Mapping[str, str],
    structure_names: Sequence[str],
    on_field_change: Callable[[str, str], None],
    on_commit: Callable[[], None],
    on_cancel: Callable[[], None],
    background_color: Optional[QtGui.QColor] = None,
) -> dict[str, QtWidgets.QWidget]:
    editor_background = background_color or QtGui.QColor(20, 20, 20)
    combo = QtWidgets.QComboBox()
    combo.addItems(list(structure_names))
    current_structure_name = constraint_editor_state.get("structure_name", "")
    if current_structure_name in structure_names:
        combo.setCurrentText(current_structure_name)

    metric_edit = QtWidgets.QLineEdit(constraint_editor_state.get("metric", ""))
    metric_edit.setPlaceholderText("e.g. V65Gy")

    goal_edit = QtWidgets.QLineEdit(constraint_editor_state.get("goal_text", ""))
    goal_edit.setPlaceholderText("e.g. <= 30 Gy")

    result_label = QtWidgets.QLabel()
    result_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
    result_label.setMargin(4)
    result_label.setStyleSheet("color: #bdbdbd;")

    for widget in (combo, metric_edit, goal_edit):
        widget.setStyleSheet(
            "QComboBox, QLineEdit {"
            f" background-color: {editor_background.lighter(118).name()};"
            " border: 1px solid #666666;"
            " padding: 2px 4px;"
            "}"
        )

    combo.currentTextChanged.connect(lambda text: on_field_change("structure_name", text))
    metric_edit.textChanged.connect(lambda text: on_field_change("metric", text))
    goal_edit.textChanged.connect(lambda text: on_field_change("goal_text", text))
    goal_edit.returnPressed.connect(on_commit)

    table.setCellWidget(row_index, 0, combo)
    table.setCellWidget(row_index, 1, metric_edit)
    table.setCellWidget(row_index, 2, goal_edit)
    table.setCellWidget(row_index, 3, result_label)
    table.setItem(
        row_index,
        4,
        create_constraint_text_item(
            "",
            editor_background,
            column_index=4,
            evaluation_status="",
            is_custom_only=False,
        ),
    )
    action_widget, add_button = create_constraint_editor_action_widget(
        editor_background,
        on_commit=on_commit,
        on_cancel=on_cancel,
    )
    table.setCellWidget(row_index, 5, action_widget)

    return {
        "structure_combo": combo,
        "metric_edit": metric_edit,
        "goal_edit": goal_edit,
        "result_label": result_label,
        "add_button": add_button,
    }
