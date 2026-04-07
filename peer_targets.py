from __future__ import annotations

import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from peer_helpers import normalize_structure_name
from peer_models import RTStructData, StructureSliceContours


def get_target_row_reference_dose_text(
    row: Mapping[str, object],
    *,
    normalize_dose_text: Callable[[str], str],
) -> str:
    stored_text = str(row.get("reference_dose_text", "")).strip()
    if stored_text:
        return stored_text
    coverage_text = str(row.get("coverage_text", ""))
    match = re.search(r"@\s*([0-9]+(?:\.[0-9]+)?)\s*Gy", coverage_text)
    if match is not None:
        return normalize_dose_text(match.group(1))
    return ""


def target_table_rows_require_recompute(
    rows: Sequence[Mapping[str, object]],
    *,
    has_ct: bool,
    has_dose: bool,
    stereotactic_summary_enabled: bool,
) -> bool:
    if not has_ct or not has_dose:
        return False
    for row in rows:
        if not bool(row.get("is_primary_ptv", False)):
            continue
        normalized_name = normalize_structure_name(str(row.get("normalized_name", "")))
        if not normalized_name.startswith("PTV"):
            continue
        coverage_text = str(row.get("coverage_text", "")).strip()
        minimum_dose_text = str(row.get("minimum_dose_text", "")).strip()
        maximum_dose_text = str(row.get("maximum_dose_text", "")).strip()
        if not coverage_text or not minimum_dose_text or not maximum_dose_text:
            return True
        if stereotactic_summary_enabled:
            notes_text = str(row.get("notes_text", "")).strip()
            if not notes_text:
                return True
    return False


def compose_target_note_text(computed_note_text: str, stored_note_text: str) -> str:
    computed = computed_note_text.strip()
    stored = stored_note_text.strip()
    if computed and stored:
        if stored == computed or stored.startswith(f"{computed}\n"):
            return stored
        return f"{computed}\n{stored}"
    if computed:
        return computed
    return stored


def build_target_notes_for_save(
    target_rows: Sequence[Mapping[str, object]],
    *,
    target_notes: Mapping[str, str],
    get_target_note_key_for_row: Callable[[Mapping[str, object]], str],
) -> Dict[str, str]:
    saved_notes: Dict[str, str] = {}
    for row in target_rows:
        note_key = get_target_note_key_for_row(row)
        combined_note_text = compose_target_note_text(
            str(row.get("notes_text", "")),
            target_notes.get(note_key, ""),
        )
        if combined_note_text:
            saved_notes[note_key] = combined_note_text

    for note_key, note_text in target_notes.items():
        cleaned_note = str(note_text).strip()
        if cleaned_note and note_key not in saved_notes:
            saved_notes[note_key] = cleaned_note
    return saved_notes


def extract_manual_target_notes(
    saved_notes_payload: Mapping[str, str],
    target_rows: Sequence[Mapping[str, object]],
    *,
    get_target_note_key_for_row: Callable[[Mapping[str, object]], str],
) -> Dict[str, str]:
    row_note_text_by_key = {
        get_target_note_key_for_row(row): str(row.get("notes_text", "")).strip()
        for row in target_rows
    }
    manual_notes: Dict[str, str] = {}
    for note_key, note_text in saved_notes_payload.items():
        cleaned_note = str(note_text).strip()
        if not cleaned_note:
            continue
        computed_note_text = row_note_text_by_key.get(note_key, "")
        if computed_note_text:
            if cleaned_note == computed_note_text:
                continue
            if cleaned_note.startswith(f"{computed_note_text}\n"):
                cleaned_note = cleaned_note[len(computed_note_text) + 1 :].strip()
        if cleaned_note:
            manual_notes[str(note_key)] = cleaned_note
    return manual_notes


def get_sorted_ptv_structures(
    rtstruct: Optional[RTStructData],
    *,
    is_listable_structure_name: Callable[[str], bool],
    parse_ptv_rx_gy_from_name: Callable[[str], Optional[float]],
) -> List[StructureSliceContours]:
    if rtstruct is None:
        return []

    def sort_key(structure: StructureSliceContours) -> Tuple[float, str]:
        rx_gy = parse_ptv_rx_gy_from_name(structure.name)
        return (rx_gy if rx_gy is not None else float("inf"), normalize_structure_name(structure.name))

    return sorted(
        [
            structure
            for structure in rtstruct.structures
            if is_listable_structure_name(normalize_structure_name(structure.name))
            and normalize_structure_name(structure.name).startswith("PTV")
        ],
        key=sort_key,
    )


def get_stereotactic_competing_ptv_entries(
    structure: StructureSliceContours,
    *,
    sorted_ptv_structures: Sequence[StructureSliceContours],
    get_target_structure_slice_masks: Callable[[StructureSliceContours], Dict[int, np.ndarray]],
    structure_is_fully_encompassed: Callable[[StructureSliceContours, StructureSliceContours], bool],
) -> List[Tuple[str, Dict[int, np.ndarray]]]:
    normalized_name = normalize_structure_name(structure.name)
    entries: List[Tuple[str, Dict[int, np.ndarray]]] = []
    for ptv_structure in sorted_ptv_structures:
        ptv_normalized_name = normalize_structure_name(ptv_structure.name)
        if ptv_normalized_name != normalized_name and (
            structure_is_fully_encompassed(structure, ptv_structure)
            or structure_is_fully_encompassed(ptv_structure, structure)
        ):
            continue
        ptv_masks = get_target_structure_slice_masks(ptv_structure)
        if not ptv_masks:
            continue
        entries.append((ptv_normalized_name, ptv_masks))
    return entries


def get_preferred_manual_target_parent_name(
    structure: StructureSliceContours,
    *,
    additional_target_subvolume_names: Sequence[str],
    sorted_ptv_structures: Sequence[StructureSliceContours],
    structure_is_fully_encompassed: Callable[[StructureSliceContours, StructureSliceContours], bool],
    get_structure_mask_voxel_count: Callable[[StructureSliceContours], int],
) -> Optional[str]:
    normalized_name = normalize_structure_name(structure.name)
    if normalized_name not in set(additional_target_subvolume_names):
        return None

    candidate_parents: List[Tuple[int, str]] = []
    for ptv_structure in sorted_ptv_structures:
        parent_normalized_name = normalize_structure_name(ptv_structure.name)
        if parent_normalized_name == normalized_name:
            continue
        if not structure_is_fully_encompassed(ptv_structure, structure):
            continue
        candidate_parents.append(
            (
                get_structure_mask_voxel_count(ptv_structure),
                parent_normalized_name,
            )
        )

    if not candidate_parents:
        return None

    _voxel_count, preferred_parent_name = min(
        candidate_parents,
        key=lambda item: (item[0], item[1]),
    )
    return preferred_parent_name


def resolve_nested_target_names(
    parent_structure: StructureSliceContours,
    *,
    rtstruct: Optional[RTStructData],
    cached_names: Optional[Sequence[str]],
    additional_target_subvolume_names: Sequence[str],
    is_listable_structure_name: Callable[[str], bool],
    is_nested_target_structure_name: Callable[[str], bool],
    parse_ptv_rx_gy_from_name: Callable[[str], Optional[float]],
    get_preferred_manual_target_parent_name: Callable[[StructureSliceContours], Optional[str]],
    structure_is_fully_encompassed: Callable[[StructureSliceContours, StructureSliceContours], bool],
) -> List[str]:
    if rtstruct is None:
        return []
    if cached_names is not None:
        return [normalize_structure_name(name) for name in cached_names]

    parent_normalized_name = normalize_structure_name(parent_structure.name)
    manual_target_names = set(additional_target_subvolume_names)

    def sort_key(structure: StructureSliceContours) -> Tuple[int, float, str]:
        normalized_name = normalize_structure_name(structure.name)
        if normalized_name.startswith("PTV"):
            type_order = 0
        elif normalized_name.startswith("GTV"):
            type_order = 1
        elif normalized_name.startswith("CTV"):
            type_order = 2
        else:
            type_order = 3
        rx_gy = parse_ptv_rx_gy_from_name(structure.name)
        return (
            type_order,
            rx_gy if rx_gy is not None else float("inf"),
            normalized_name,
        )

    nested_structures: List[StructureSliceContours] = []
    for structure in rtstruct.structures:
        normalized_name = normalize_structure_name(structure.name)
        if normalized_name == parent_normalized_name:
            continue
        if not is_listable_structure_name(normalized_name):
            continue
        if normalized_name in manual_target_names:
            preferred_parent_name = get_preferred_manual_target_parent_name(structure)
            if preferred_parent_name != parent_normalized_name:
                continue
        elif not is_nested_target_structure_name(normalized_name):
            continue
        if structure_is_fully_encompassed(parent_structure, structure):
            nested_structures.append(structure)

    nested_structures.sort(key=sort_key)
    return [normalize_structure_name(structure.name) for structure in nested_structures]


def localize_stereotactic_extra_mask(
    context: Mapping[str, object],
    extra_mask: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if extra_mask is None:
        return None

    z_start = int(context["z_start"])
    z_end = int(context["z_end"])
    row_start = int(context["row_start"])
    row_end = int(context["row_end"])
    col_start = int(context["col_start"])
    col_end = int(context["col_end"])
    return np.asarray(
        extra_mask[
            z_start: z_end + 1,
            row_start: row_end + 1,
            col_start: col_end + 1,
        ],
        dtype=bool,
    )


def compute_stereotactic_owned_volume_cc(
    context: Mapping[str, object],
    threshold_gy: float,
    *,
    extra_mask: Optional[np.ndarray] = None,
) -> float:
    if threshold_gy <= 0.0:
        return 0.0

    target_weight = np.asarray(context["target_weight"], dtype=np.float32)
    dose_block = np.asarray(context["dose_block"], dtype=np.float32)
    voxel_volume_cc = float(context["voxel_volume_cc"])

    isodose_mask = np.asarray(dose_block >= float(threshold_gy), dtype=bool)
    localized_extra_mask = localize_stereotactic_extra_mask(context, extra_mask)
    if localized_extra_mask is not None:
        isodose_mask &= localized_extra_mask
    if not np.any(isodose_mask):
        return 0.0
    return float(np.sum(target_weight[isodose_mask]) * voxel_volume_cc)
