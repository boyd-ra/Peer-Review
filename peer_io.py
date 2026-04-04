from __future__ import annotations
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
from openpyxl import load_workbook

from peer_helpers import (
    get_ct_row_col_normal,
    get_ipp,
    get_iop,
    normalize_structure_name,
    safe_get,
)
from peer_models import CTVolume, DoseVolume, RTPlanPhase, RTStructData, StructureGoal, StructureSliceContours

try:
    from pydicom.pixels import apply_rescale
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_modality_lut as apply_rescale


logger = logging.getLogger(__name__)


def get_constraints_workbook_path() -> Optional[str]:
    path = Path(__file__).resolve().with_name("constraints.xlsx")
    if path.exists():
        return str(path)
    return None


def _parse_structure_goal_rows(
    fieldnames: List[str],
    rows: List[Dict[str, object]],
) -> Tuple[set[str], dict[str, List[StructureGoal]], List[str]]:
    allowed_names: set[str] = set()
    goals_by_structure: dict[str, List[StructureGoal]] = {}
    structure_order: List[str] = []
    oar_field = next((field for field in fieldnames if normalize_structure_name(field) == "OAR"), None)
    metric_field = next((field for field in fieldnames if normalize_structure_name(field) == "METRIC"), None)
    goal_field = next((field for field in fieldnames if normalize_structure_name(field) == "GOAL"), None)
    value_field = next((field for field in fieldnames if normalize_structure_name(field) == "VALUE"), None)
    fallback_field = fieldnames[0] if fieldnames else None

    def unpack_cell(cell_payload: object) -> Tuple[object, str]:
        if isinstance(cell_payload, dict):
            return cell_payload.get("value"), str(cell_payload.get("number_format", "") or "")
        return cell_payload, ""

    def format_numeric_text(value: float) -> str:
        rounded = round(value)
        if np.isclose(value, rounded):
            return str(int(rounded))
        return f"{value:.6g}"

    def format_value_text(cell_payload: object, metric_text: str) -> str:
        raw_value, number_format = unpack_cell(cell_payload)
        if raw_value is None:
            return ""
        if isinstance(raw_value, str):
            return raw_value.strip()
        if isinstance(raw_value, (int, float, np.integer, np.floating)):
            numeric_value = float(raw_value)
            metric_key = metric_text.strip().upper().replace(" ", "")
            is_percent_value = "%" in number_format or (
                metric_key.startswith("V") and 0.0 <= numeric_value <= 1.0
            )
            if is_percent_value:
                return f"{format_numeric_text(numeric_value * 100.0)}%"
            return format_numeric_text(numeric_value)
        return str(raw_value).strip()

    def split_structure_names(raw_name_text: str) -> List[str]:
        split_names = [part.strip() for part in re.split(r",", raw_name_text) if part.strip()]
        expanded_names: List[str] = []
        for name in split_names:
            match = re.fullmatch(r"(.+)_L/R", name, flags=re.IGNORECASE)
            if match is not None:
                prefix = match.group(1).strip()
                if prefix:
                    expanded_names.extend([f"{prefix}_L", f"{prefix}_R"])
                continue
            expanded_names.append(name)
        if expanded_names:
            return expanded_names
        stripped = raw_name_text.strip()
        return [stripped] if stripped else []

    for row in rows:
        raw_name = ""
        if oar_field:
            raw_name, _ = unpack_cell(row.get(oar_field, ""))
            raw_name = str(raw_name or "")
        elif fallback_field:
            raw_name, _ = unpack_cell(row.get(fallback_field, ""))
            raw_name = str(raw_name or "")
        structure_names = split_structure_names(raw_name)
        if structure_names:
            metric_value, _ = unpack_cell(row.get(metric_field, "")) if metric_field else ("", "")
            comparator_value, _ = unpack_cell(row.get(goal_field, "")) if goal_field else ("", "")
            metric = str(metric_value or "").strip()
            comparator = str(comparator_value or "").strip()
            value_text = format_value_text(row.get(value_field, ""), metric) if value_field else ""
            for structure_name in structure_names:
                normalized = normalize_structure_name(structure_name)
                if not normalized:
                    continue
                if normalized not in allowed_names:
                    structure_order.append(normalized)
                allowed_names.add(normalized)
                if metric or comparator or value_text:
                    goals_by_structure.setdefault(normalized, []).append(
                        StructureGoal(
                            structure_name=structure_name,
                            metric=metric,
                            comparator=comparator,
                            value_text=value_text,
                        )
                    )

    return allowed_names, goals_by_structure, structure_order


def list_constraints_workbook_sheets(path: str) -> List[str]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        return list(workbook.sheetnames)
    finally:
        workbook.close()


def _cell_payload(cell) -> Dict[str, object]:
    return {
        "value": cell.value,
        "number_format": cell.number_format,
    }


def _cell_text(cell_payload: object) -> str:
    if isinstance(cell_payload, dict):
        value = cell_payload.get("value")
    else:
        value = cell_payload
    if value is None:
        return ""
    return str(value).strip()


def _normalize_constraint_header(cell_payload: object) -> str:
    return normalize_structure_name(_cell_text(cell_payload))


def _extract_plan_dose_per_fraction_values(plan_phases: Optional[List[RTPlanPhase]]) -> List[float]:
    values: List[float] = []
    if not plan_phases:
        return values
    for phase in plan_phases:
        if phase.prescription_dose_gy <= 0.0 or phase.fractions_planned <= 0:
            continue
        dose_per_fraction_gy = phase.prescription_dose_gy / float(phase.fractions_planned)
        if any(abs(existing - dose_per_fraction_gy) <= 0.01 for existing in values):
            continue
        values.append(dose_per_fraction_gy)
    return values


def _extract_plan_fraction_counts(plan_phases: Optional[List[RTPlanPhase]]) -> List[int]:
    values: List[int] = []
    if not plan_phases:
        return values
    for phase in plan_phases:
        fractions_planned = int(phase.fractions_planned)
        if fractions_planned <= 0:
            continue
        if fractions_planned in values:
            continue
        values.append(fractions_planned)
    return values


def _parse_constraint_block_label(label_text: str) -> Tuple[str, Optional[float]]:
    stripped = label_text.strip()
    if not stripped:
        return "", None
    if normalize_structure_name(stripped) in {"NA", "N/A"}:
        return "na", None
    match = re.search(r"\bD\s*/\s*F\b\s*([0-9]+(?:\.[0-9]+)?)", stripped, flags=re.IGNORECASE)
    if match is not None:
        try:
            return "dose_per_fraction", float(match.group(1))
        except ValueError:
            return "", None
    match = re.search(r"\bF\b\s*([0-9]+(?:\.[0-9]+)?)", stripped, flags=re.IGNORECASE)
    if match is not None:
        try:
            return "fraction_count", float(match.group(1))
        except ValueError:
            return "", None
    return "", None


def _extract_constraints_table_blocks(worksheet) -> List[Dict[str, object]]:
    rows = list(worksheet.iter_rows(values_only=False))
    if len(rows) < 2:
        return []

    max_cols = max(len(row) for row in rows)
    label_row_payloads = [_cell_payload(cell) for cell in rows[0]]
    header_row_payloads = [_cell_payload(cell) for cell in rows[1]]

    blocks: List[Dict[str, object]] = []
    col = 0
    while col + 3 < max_cols:
        headers = [
            _normalize_constraint_header(header_row_payloads[col + offset] if col + offset < len(header_row_payloads) else None)
            for offset in range(4)
        ]
        if headers == ["OAR", "METRIC", "GOAL", "VALUE"]:
            fieldnames = [
                _cell_text(header_row_payloads[col + offset] if col + offset < len(header_row_payloads) else None)
                for offset in range(4)
            ]
            block_rows: List[Dict[str, object]] = []
            for row in rows[2:]:
                payloads = [_cell_payload(cell) for cell in row]
                row_payload: Dict[str, object] = {}
                has_any_value = False
                for offset, fieldname in enumerate(fieldnames):
                    payload = payloads[col + offset] if col + offset < len(payloads) else None
                    row_payload[fieldname] = payload
                    if _cell_text(payload):
                        has_any_value = True
                if has_any_value:
                    block_rows.append(row_payload)
            label_payload = label_row_payloads[col] if col < len(label_row_payloads) else None
            blocks.append(
                {
                    "label": _cell_text(label_payload),
                    "fieldnames": fieldnames,
                    "rows": block_rows,
                }
            )
            col += 4
            continue
        col += 1

    return blocks


def _select_constraints_table_block(
    blocks: List[Dict[str, object]],
    plan_phases: Optional[List[RTPlanPhase]],
) -> Optional[Dict[str, object]]:
    if not blocks:
        return None
    if len(blocks) == 1:
        return blocks[0]

    dose_per_fraction_values = _extract_plan_dose_per_fraction_values(plan_phases)
    fraction_counts = _extract_plan_fraction_counts(plan_phases)
    fallback_block: Optional[Dict[str, object]] = None
    matching_blocks: List[Tuple[float, Dict[str, object]]] = []

    for block in blocks:
        label_kind, label_value = _parse_constraint_block_label(str(block.get("label", "")))
        if label_kind == "na":
            fallback_block = block
        elif label_kind == "dose_per_fraction" and label_value is not None:
            for dose_per_fraction_gy in dose_per_fraction_values:
                if abs(label_value - dose_per_fraction_gy) <= 0.05:
                    matching_blocks.append((abs(label_value - dose_per_fraction_gy), block))
                    break
        elif label_kind == "fraction_count" and label_value is not None:
            for fraction_count in fraction_counts:
                if abs(label_value - float(fraction_count)) <= 0.05:
                    matching_blocks.append((abs(label_value - float(fraction_count)), block))
                    break

    if matching_blocks:
        matching_blocks.sort(key=lambda item: item[0])
        return matching_blocks[0][1]
    if fallback_block is not None:
        return fallback_block
    return blocks[0]


def load_structure_constraints_sheet(
    path: str,
    sheet_name: str,
    plan_phases: Optional[List[RTPlanPhase]] = None,
) -> Tuple[set[str], dict[str, List[StructureGoal]], List[str]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            raise ValueError(f"Constraints sheet '{sheet_name}' was not found in {Path(path).name}.")
        worksheet = workbook[sheet_name]
        blocks = _extract_constraints_table_blocks(worksheet)
        selected_block = _select_constraints_table_block(blocks, plan_phases)
        if selected_block is not None:
            return _parse_structure_goal_rows(
                list(selected_block.get("fieldnames", [])),
                list(selected_block.get("rows", [])),
            )

        row_iter = worksheet.iter_rows(values_only=False)
        header_row = next(row_iter, None)
        if header_row is None:
            return set(), {}, []
        fieldnames = [str(cell.value).strip() if cell.value is not None else "" for cell in header_row]
        rows: List[Dict[str, object]] = []
        for cells in row_iter:
            row: Dict[str, object] = {}
            for index, fieldname in enumerate(fieldnames):
                if index < len(cells):
                    row[fieldname] = _cell_payload(cells[index])
                else:
                    row[fieldname] = None
            rows.append(row)
        return _parse_structure_goal_rows(fieldnames, rows)
    finally:
        workbook.close()


def scan_patient_folder(
    folder: str,
) -> Tuple[List[str], Optional[str], List[str], List[str]]:
    rtstruct_paths: List[str] = []
    rtdose_paths: List[str] = []
    rtplan_paths: List[str] = []
    ct_paths: List[str] = []

    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file():
            continue

        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            modality = str(safe_get(ds, "Modality", "")).upper()
        except Exception as exc:
            logger.debug("Skipping unreadable DICOM during folder scan: %s (%s)", path, exc)
            continue

        if modality == "CT":
            ct_paths.append(str(path))
        elif modality == "RTSTRUCT":
            rtstruct_paths.append(str(path))
        elif modality == "RTDOSE":
            rtdose_paths.append(str(path))
        elif modality == "RTPLAN":
            rtplan_paths.append(str(path))

    rtstruct_path = rtstruct_paths[0] if rtstruct_paths else None
    return ct_paths, rtstruct_path, rtdose_paths, rtplan_paths

def load_ct_series_from_paths(ct_paths: List[str]) -> CTVolume:
    files = []
    for path in ct_paths:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)
            if safe_get(ds, "Modality", "") == "CT":
                files.append(ds)
        except Exception as exc:
            logger.debug("Skipping CT candidate during CT load: %s (%s)", path, exc)

    if not files:
        raise ValueError("No CT DICOM slices found in the selected folder.")

    first_iop = get_iop(files[0])
    row_cos, col_cos, normal = get_ct_row_col_normal(first_iop)

    def slice_sort_key(ds):
        ipp = get_ipp(ds)
        return float(np.dot(ipp, normal))

    files.sort(key=slice_sort_key)

    first = files[0]
    rows = int(first.Rows)
    cols = int(first.Columns)

    px_spacing = np.array([float(x) for x in first.PixelSpacing], dtype=float)
    sy = float(px_spacing[0])
    sx = float(px_spacing[1])

    slice_origins = []
    slice_positions_along_normal = []
    slices = []

    for ds in files:
        arr = ds.pixel_array.astype(np.float32)
        arr = apply_rescale(arr, ds).astype(np.float32)
        slices.append(arr)

        ipp = get_ipp(ds)
        slice_origins.append(ipp)
        slice_positions_along_normal.append(float(np.dot(ipp, normal)))

    volume = np.stack(slices, axis=0)
    slice_origins = np.asarray(slice_origins, dtype=float)
    slice_positions_along_normal = np.asarray(slice_positions_along_normal, dtype=float)

    if len(slice_positions_along_normal) > 1:
        dz = float(np.median(np.diff(slice_positions_along_normal)))
    else:
        dz = float(safe_get(first, "SliceThickness", 1.0))

    return CTVolume(
        volume_hu=volume,
        slice_origins_xyz_mm=slice_origins,
        z_positions_mm=slice_positions_along_normal,
        spacing_xyz_mm=np.array([sx, sy, abs(dz)], dtype=float),
        image_orientation_patient=first_iop,
        study_uid=str(safe_get(first, "StudyInstanceUID", "")),
        frame_of_reference_uid=str(safe_get(first, "FrameOfReferenceUID", "")),
        rows=rows,
        cols=cols,
    )


def load_ct_series(folder: str) -> CTVolume:
    ct_paths, _, _, _ = scan_patient_folder(folder)
    return load_ct_series_from_paths(ct_paths)


def load_ct_series_and_discover_patient_files(
    folder: str,
) -> Tuple[CTVolume, Optional[str], List[str], List[str]]:
    ct_paths, rtstruct_path, rtdose_paths, rtplan_paths = scan_patient_folder(folder)
    return load_ct_series_from_paths(ct_paths), rtstruct_path, rtdose_paths, rtplan_paths


def _format_patient_name(name_value: object) -> str:
    text = str(name_value or "").strip()
    if not text:
        return ""
    return " ".join(part for part in text.replace("^", " ").split() if part)


def _extract_rtplan_prescription_doses_gy(ds: pydicom.dataset.Dataset) -> List[float]:
    prescription_doses: List[float] = []
    for item in safe_get(ds, "DoseReferenceSequence", []):
        value = safe_get(item, "TargetPrescriptionDose", None)
        if value in {None, ""}:
            continue
        try:
            prescription_doses.append(float(value))
        except (TypeError, ValueError):
            continue

    return prescription_doses


def _extract_rtplan_number_of_fractions(ds: pydicom.dataset.Dataset) -> int:
    total_fractions = 0
    for item in safe_get(ds, "FractionGroupSequence", []):
        value = safe_get(item, "NumberOfFractionsPlanned", None)
        if value in {None, ""}:
            continue
        try:
            total_fractions += int(value)
        except (TypeError, ValueError):
            continue

    return total_fractions


def _extract_rtplan_target_structure_name(ds: pydicom.dataset.Dataset) -> str:
    for item in safe_get(ds, "DoseReferenceSequence", []):
        description = str(safe_get(item, "DoseReferenceDescription", "")).strip()
        if not description:
            continue
        normalized_description = normalize_structure_name(description)
        ptv_index = normalized_description.find("PTV")
        if ptv_index >= 0:
            return normalized_description[ptv_index:]
    return ""


def load_rtplan_phases(paths: List[str], rtdose_paths: Optional[List[str]] = None) -> List[RTPlanPhase]:
    rtdose_paths = rtdose_paths or []
    dose_path_by_plan_uid: Dict[str, str] = {}

    for path in sorted(rtdose_paths):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception as exc:
            logger.debug("Skipping RTDOSE while loading RTPLAN phases: %s (%s)", path, exc)
            continue
        if str(safe_get(ds, "Modality", "")).upper() != "RTDOSE":
            continue
        for item in safe_get(ds, "ReferencedRTPlanSequence", []):
            referenced_uid = str(safe_get(item, "ReferencedSOPInstanceUID", "")).strip()
            if referenced_uid and referenced_uid not in dose_path_by_plan_uid:
                dose_path_by_plan_uid[referenced_uid] = path

    phases: List[RTPlanPhase] = []
    for path in sorted(paths):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception as exc:
            logger.debug("Skipping RTPLAN while loading phases: %s (%s)", path, exc)
            continue
        if str(safe_get(ds, "Modality", "")).upper() != "RTPLAN":
            continue

        sop_instance_uid = str(safe_get(ds, "SOPInstanceUID", "")).strip()
        prescription_doses_gy = _extract_rtplan_prescription_doses_gy(ds)
        phases.append(
            RTPlanPhase(
                sop_instance_uid=sop_instance_uid,
                prescription_dose_gy=max(prescription_doses_gy) if prescription_doses_gy else 0.0,
                fractions_planned=_extract_rtplan_number_of_fractions(ds),
                dose_path=dose_path_by_plan_uid.get(sop_instance_uid, ""),
                target_structure_name=_extract_rtplan_target_structure_name(ds),
                plan_label=str(safe_get(ds, "RTPlanLabel", "")).strip(),
                plan_name=str(safe_get(ds, "RTPlanName", "")).strip(),
            )
        )

    return phases


def summarize_rtplan_files(paths: List[str]) -> Optional[Tuple[str, ...]]:
    if not paths:
        return None

    patient_name = ""
    patient_id = ""
    total_prescription_dose_gy = 0.0
    total_fractions = 0
    plan_count = 0

    for path in sorted(paths):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception as exc:
            logger.debug("Skipping RTPLAN while summarizing plans: %s (%s)", path, exc)
            continue
        if str(safe_get(ds, "Modality", "")).upper() != "RTPLAN":
            continue

        plan_count += 1
        if not patient_name:
            patient_name = _format_patient_name(safe_get(ds, "PatientName", ""))
        if not patient_id:
            patient_id = str(safe_get(ds, "PatientID", "")).strip()

        plan_prescription_doses_gy = _extract_rtplan_prescription_doses_gy(ds)
        if plan_prescription_doses_gy:
            total_prescription_dose_gy += max(plan_prescription_doses_gy)
        total_fractions += _extract_rtplan_number_of_fractions(ds)

    if not patient_name and not patient_id and total_prescription_dose_gy <= 0.0 and total_fractions <= 0 and plan_count <= 0:
        return None

    line_1 = patient_name or "Patient name unavailable"
    line_2 = f"ID: {patient_id}" if patient_id else "ID unavailable"
    if total_fractions > 0:
        dose_per_fraction_gy = total_prescription_dose_gy / float(total_fractions)
        line_3 = f"{total_prescription_dose_gy:.2f} Gy | {total_fractions} fx | {dose_per_fraction_gy:.2f} Gy/fx"
    elif total_prescription_dose_gy > 0.0:
        line_3 = f"{total_prescription_dose_gy:.2f} Gy | Fractions unavailable"
    else:
        line_3 = "Prescription unavailable"

    if plan_count > 0:
        phase_label = "phase" if plan_count == 1 else "phases"
        return line_1, line_2, line_3, f"{plan_count} {phase_label}"

    return line_1, line_2, line_3


def load_rtdose(path: str) -> DoseVolume:
    ds = pydicom.dcmread(path, stop_before_pixels=False)
    if safe_get(ds, "Modality", "") != "RTDOSE":
        raise ValueError("Selected file is not an RTDOSE object.")

    arr = ds.pixel_array.astype(np.float32)
    dose_grid_scaling = float(safe_get(ds, "DoseGridScaling", 1.0))
    arr = arr * dose_grid_scaling

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    px_spacing = np.array([float(x) for x in ds.PixelSpacing], dtype=float)
    sy = float(px_spacing[0])
    sx = float(px_spacing[1])

    iop = get_iop(ds)
    _, _, dose_normal = get_ct_row_col_normal(iop)
    ipp = get_ipp(ds)
    offsets = np.array(safe_get(ds, "GridFrameOffsetVector", list(range(arr.shape[0]))), dtype=float)
    slice_origins = ipp[None, :] + offsets[:, None] * dose_normal[None, :]
    z_positions = slice_origins @ dose_normal

    dz = float(np.median(np.diff(z_positions))) if len(z_positions) > 1 else 1.0

    return DoseVolume(
        dose_gy=arr,
        slice_origins_xyz_mm=slice_origins,
        z_positions_mm=z_positions,
        origin_xyz_mm=slice_origins[0].copy(),
        spacing_xyz_mm=np.array([sx, sy, abs(dz)], dtype=float),
        image_orientation_patient=iop,
        frame_of_reference_uid=str(safe_get(ds, "FrameOfReferenceUID", "")),
        dose_units=str(safe_get(ds, "DoseUnits", "")),
    )


def validate_dose_geometry(reference: DoseVolume, candidate: DoseVolume, path: str):
    if reference.dose_gy.shape != candidate.dose_gy.shape:
        raise ValueError(
            f"RTDOSE file '{path}' does not match the reference dose grid shape "
            f"{reference.dose_gy.shape} != {candidate.dose_gy.shape}."
        )
    if reference.slice_origins_xyz_mm.shape != candidate.slice_origins_xyz_mm.shape:
        raise ValueError(
            f"RTDOSE file '{path}' does not match the reference dose origins shape "
            f"{reference.slice_origins_xyz_mm.shape} != {candidate.slice_origins_xyz_mm.shape}."
        )

    checks = [
        (np.allclose(reference.slice_origins_xyz_mm, candidate.slice_origins_xyz_mm, atol=1e-3), "dose origins"),
        (np.allclose(reference.spacing_xyz_mm, candidate.spacing_xyz_mm, atol=1e-6), "dose spacing"),
        (
            np.allclose(reference.image_orientation_patient, candidate.image_orientation_patient, atol=1e-6),
            "dose orientation",
        ),
    ]
    for passed, label in checks:
        if not passed:
            raise ValueError(f"RTDOSE file '{path}' does not match the reference {label}.")


def load_combined_rtdose(paths: List[str]) -> DoseVolume:
    if not paths:
        raise ValueError("No RTDOSE files were provided.")

    loaded = [load_rtdose(path) for path in sorted(paths)]
    reference = loaded[0]
    combined_dose = reference.dose_gy.copy()

    for path, dose in zip(sorted(paths)[1:], loaded[1:]):
        validate_dose_geometry(reference, dose, path)
        combined_dose += dose.dose_gy

    return DoseVolume(
        dose_gy=combined_dose,
        slice_origins_xyz_mm=reference.slice_origins_xyz_mm.copy(),
        z_positions_mm=reference.z_positions_mm.copy(),
        origin_xyz_mm=reference.origin_xyz_mm.copy(),
        spacing_xyz_mm=reference.spacing_xyz_mm.copy(),
        image_orientation_patient=reference.image_orientation_patient.copy(),
        frame_of_reference_uid=reference.frame_of_reference_uid,
        dose_units=reference.dose_units,
    )


def load_rtstruct(path: str, ct: CTVolume) -> RTStructData:
    ds = pydicom.dcmread(path, stop_before_pixels=False)
    if safe_get(ds, "Modality", "") != "RTSTRUCT":
        raise ValueError("Selected file is not an RTSTRUCT object.")

    # The viewer keeps all RTSTRUCT entries so the axial and DVH tabs can
    # decide independently which structures to show or compute.
    row_cos, col_cos, normal = get_ct_row_col_normal(ct.image_orientation_patient)
    inv_sx = 1.0 / max(float(ct.spacing_xyz_mm[0]), 1e-6)
    inv_sy = 1.0 / max(float(ct.spacing_xyz_mm[1]), 1e-6)
    slice_origins = np.asarray(ct.slice_origins_xyz_mm, dtype=np.float32)
    z_positions = np.asarray(ct.z_positions_mm, dtype=np.float32)

    if z_positions.size > 1:
        z_steps = np.diff(z_positions)
        nominal_dz = float(np.median(z_steps))
        use_direct_slice_lookup = abs(nominal_dz) > 1e-6 and np.allclose(z_steps, nominal_dz, atol=1e-3)
    else:
        nominal_dz = 0.0
        use_direct_slice_lookup = False

    def nearest_slice_index_for_contour(contour_xyz: np.ndarray) -> int:
        contour_pos = float(np.mean(contour_xyz @ normal))
        if use_direct_slice_lookup:
            approx_index = int(round((contour_pos - float(z_positions[0])) / nominal_dz))
            return int(np.clip(approx_index, 0, len(z_positions) - 1))

        idx = int(np.searchsorted(z_positions, contour_pos, side="left"))
        if idx <= 0:
            return 0
        if idx >= len(z_positions):
            return len(z_positions) - 1
        if abs(contour_pos - float(z_positions[idx - 1])) <= abs(float(z_positions[idx]) - contour_pos):
            return idx - 1
        return idx

    roi_name_by_number = {}
    for item in safe_get(ds, "StructureSetROISequence", []):
        roi_name_by_number[int(item.ROINumber)] = str(item.ROIName)

    structures: List[StructureSliceContours] = []

    for roi_contour in safe_get(ds, "ROIContourSequence", []):
        roi_num = int(roi_contour.ReferencedROINumber)
        name = roi_name_by_number.get(roi_num, f"ROI {roi_num}")
        color = tuple(int(c) for c in safe_get(roi_contour, "ROIDisplayColor", [255, 255, 0]))
        by_slice = {}

        for contour in safe_get(roi_contour, "ContourSequence", []):
            data = np.asarray(contour.ContourData, dtype=np.float32).reshape(-1, 3)
            if len(data) < 3:
                continue

            k = nearest_slice_index_for_contour(data)
            rel = data - slice_origins[k][None, :]
            cols = rel @ row_cos * inv_sx
            rows = rel @ col_cos * inv_sy
            rc = np.column_stack([rows, cols]).astype(np.float32, copy=False)
            by_slice.setdefault(k, []).append(rc)

        structures.append(
            StructureSliceContours(
                name=name,
                color_rgb=color,
                points_rc_by_slice=by_slice,
            )
        )

    return RTStructData(
        structures=structures,
        frame_of_reference_uid=str(safe_get(ds, "FrameOfReferenceUID", "")),
    )
