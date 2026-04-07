from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class TargetTablePresentationRow:
    structure_name: str
    normalized_name: str
    parent_structure_name: Optional[str]
    display_name: str
    coverage_text: str
    minimum_dose_text: str
    maximum_dose_text: str
    reference_dose_text: str
    note_key: str
    note_title: str
    note_text: str
    is_primary_ptv: bool
    color_rgb: Tuple[int, int, int]


def get_target_table_column_widths(viewport_width: int) -> list[int]:
    viewport_width = max(int(viewport_width), 1)
    ptv_width = max(1, int(round(viewport_width * 0.15)))
    coverage_width = max(1, int(round(viewport_width * 0.15)))
    min_dose_width = max(1, int(round(viewport_width * 0.075)))
    max_dose_width = max(1, int(round(viewport_width * 0.075)))
    note_button_width = max(72, int(round(viewport_width * 0.06)))
    notes_width = max(
        1,
        viewport_width - (ptv_width + coverage_width + min_dose_width + max_dose_width + note_button_width),
    )
    return [
        ptv_width,
        coverage_width,
        min_dose_width,
        max_dose_width,
        notes_width,
        note_button_width,
    ]


def build_target_table_presentation_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    target_notes: Mapping[str, str],
    get_target_note_key_for_row: Callable[[Mapping[str, object]], str],
    compose_target_note_text: Callable[[str, str], str],
    get_target_row_reference_dose_text: Callable[[Mapping[str, object]], str],
) -> list[TargetTablePresentationRow]:
    presentation_rows: list[TargetTablePresentationRow] = []
    for row in rows:
        structure_name = str(row.get("structure_name", ""))
        parent_structure_name_value = row.get("parent_structure_name")
        parent_structure_name = (
            str(parent_structure_name_value)
            if parent_structure_name_value is not None
            else None
        )
        is_primary_ptv = bool(row.get("is_primary_ptv", False))
        note_key = get_target_note_key_for_row(row)
        if is_primary_ptv:
            note_title = f"{structure_name} target review"
        else:
            note_title = f"{structure_name} within {parent_structure_name} target review"
        stored_note_text = target_notes.get(note_key, "").strip()
        computed_note_text = str(row.get("notes_text", "")).strip()
        note_text = compose_target_note_text(computed_note_text, stored_note_text)
        color_values = row.get("color_rgb", (255, 255, 255))
        color_rgb = tuple(int(value) for value in color_values)
        presentation_rows.append(
            TargetTablePresentationRow(
                structure_name=structure_name,
                normalized_name=str(row.get("normalized_name", "")),
                parent_structure_name=parent_structure_name,
                display_name=str(row.get("display_name", structure_name)),
                coverage_text=str(row.get("coverage_text", "")),
                minimum_dose_text=str(row.get("minimum_dose_text", "")),
                maximum_dose_text=str(row.get("maximum_dose_text", "")),
                reference_dose_text=get_target_row_reference_dose_text(row),
                note_key=note_key,
                note_title=note_title,
                note_text=note_text,
                is_primary_ptv=is_primary_ptv,
                color_rgb=color_rgb,  # type: ignore[arg-type]
            )
        )
    return presentation_rows


def create_target_name_cell_widget(
    display_name: str,
    color_rgb: Tuple[int, int, int],
    background_color: QtGui.QColor,
    *,
    font_point_size: int,
    is_primary_ptv: bool = False,
) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget()
    widget.setStyleSheet(f"background-color: {background_color.name()};")
    layout = QtWidgets.QHBoxLayout(widget)
    layout.setContentsMargins(6, 0, 6, 0)
    layout.setSpacing(6)

    label = QtWidgets.QLabel(display_name)
    label.setStyleSheet(f"color: rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]});")
    font = label.font()
    font.setPointSize(font_point_size)
    if is_primary_ptv:
        font.setBold(True)
    label.setFont(font)
    layout.addWidget(label)
    layout.addStretch(1)
    return widget


def create_target_coverage_cell_widget(
    coverage_text: str,
    background_color: QtGui.QColor,
    *,
    structure_name: str,
    normalized_name: str,
    is_primary_ptv: bool,
    resolved_dose_text: Optional[str],
    fallback_dose_text: str,
    font_point_size: int,
    on_editing_finished: Callable[[str, str, QtWidgets.QLineEdit], None],
) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget()
    widget.setStyleSheet(f"background-color: {background_color.name()};")
    layout = QtWidgets.QHBoxLayout(widget)
    layout.setContentsMargins(6, 0, 6, 0)
    layout.setSpacing(6)

    if not is_primary_ptv:
        label = QtWidgets.QLabel(coverage_text)
        label.setStyleSheet("color: #f2f2f2;")
        label_font = label.font()
        label_font.setPointSize(font_point_size)
        label.setFont(label_font)
        layout.addWidget(label)
        layout.addStretch(1)
        return widget

    prefix_text = coverage_text.strip()
    if "@" in prefix_text:
        before_at, _sep, _after_at = prefix_text.partition("@")
        prefix_text = f"{before_at.strip()} @" if before_at.strip() else "@"

    if prefix_text:
        label = QtWidgets.QLabel(prefix_text)
        label.setStyleSheet("color: #f2f2f2;")
        label_font = label.font()
        label_font.setPointSize(font_point_size)
        label.setFont(label_font)
        layout.addWidget(label)

    dose_edit = QtWidgets.QLineEdit()
    dose_edit.setFixedWidth(62)
    dose_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    dose_edit.setPlaceholderText("Dose")
    dose_edit.setText((resolved_dose_text or fallback_dose_text).strip())
    dose_edit.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
    dose_validator = QtGui.QDoubleValidator(0.0, 999.99, 2, dose_edit)
    dose_validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
    dose_edit.setValidator(dose_validator)
    dose_font = dose_edit.font()
    dose_font.setPointSize(font_point_size)
    dose_edit.setFont(dose_font)
    dose_edit.editingFinished.connect(
        lambda name=normalized_name, structure=structure_name, edit=dose_edit: on_editing_finished(
            name,
            structure,
            edit,
        )
    )
    layout.addWidget(dose_edit, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
    layout.addStretch(1)
    return widget


def create_target_text_item(
    text: str,
    background_color: QtGui.QColor,
    *,
    tooltip_text: Optional[str] = None,
    top_align: bool = False,
) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
    item.setBackground(background_color)
    item.setForeground(QtGui.QColor("#f2f2f2"))
    if top_align and text:
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
    if tooltip_text:
        item.setToolTip(tooltip_text)
    return item


def create_target_note_button_widget(
    note_key: str,
    title: str,
    note_text: str,
    background_color: Optional[QtGui.QColor],
    *,
    on_edit_note: Callable[[str, str], None],
) -> QtWidgets.QWidget:
    button = QtWidgets.QPushButton("Note")
    button.setFixedWidth(56)
    button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
    button.setAutoDefault(False)
    button.setDefault(False)
    cleaned_note_text = note_text.strip()
    if cleaned_note_text:
        font = button.font()
        font.setBold(True)
        button.setFont(font)
        button.setToolTip(cleaned_note_text)
    else:
        button.setToolTip(f"Add note for {title}")
    if background_color is not None:
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
    if background_color is not None:
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


def populate_target_table_rows(
    table: QtWidgets.QTableWidget,
    presentation_rows: Sequence[TargetTablePresentationRow],
    *,
    primary_background: QtGui.QColor,
    nested_background: QtGui.QColor,
    get_fallback_dose_text: Callable[[str, str], str],
    on_editing_finished: Callable[[str, str, QtWidgets.QLineEdit], None],
    on_edit_note: Callable[[str, str], None],
) -> None:
    table.setRowCount(len(presentation_rows))
    font_point_size = table.font().pointSize()
    for row_index, row in enumerate(presentation_rows):
        background_color = primary_background if row.is_primary_ptv else nested_background

        table.setCellWidget(
            row_index,
            0,
            create_target_name_cell_widget(
                row.display_name,
                row.color_rgb,
                background_color,
                font_point_size=font_point_size,
                is_primary_ptv=row.is_primary_ptv,
            ),
        )

        table.setCellWidget(
            row_index,
            1,
            create_target_coverage_cell_widget(
                row.coverage_text,
                background_color,
                structure_name=row.structure_name,
                normalized_name=row.normalized_name,
                is_primary_ptv=row.is_primary_ptv,
                resolved_dose_text=row.reference_dose_text,
                fallback_dose_text=get_fallback_dose_text(row.normalized_name, row.structure_name),
                font_point_size=font_point_size,
                on_editing_finished=on_editing_finished,
            ),
        )

        for column_index, text in enumerate((row.minimum_dose_text, row.maximum_dose_text, row.note_text), start=2):
            table.setItem(
                row_index,
                column_index,
                create_target_text_item(
                    text,
                    background_color,
                    tooltip_text=row.note_text if column_index == 4 and row.note_text else None,
                    top_align=column_index == 4,
                ),
            )

        table.setCellWidget(
            row_index,
            5,
            create_target_note_button_widget(
                row.note_key,
                row.note_title,
                row.note_text,
                background_color,
                on_edit_note=on_edit_note,
            ),
        )
