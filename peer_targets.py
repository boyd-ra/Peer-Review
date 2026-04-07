from __future__ import annotations

import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from peer_helpers import normalize_structure_name
from peer_models import RTPlanPhase, RTStructData, StructureSliceContours


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


def stereotactic_summary_enabled(constraints_sheet_name: str) -> bool:
    return normalize_structure_name(constraints_sheet_name or "") == "SRS FSRT"


def _get_available_phases(plan_phases: Sequence[RTPlanPhase]) -> List[RTPlanPhase]:
    return [
        phase
        for phase in plan_phases
        if phase.prescription_dose_gy > 0.0 and phase.dose_path
    ]


def get_target_fraction_count(
    normalized_name: str,
    source_key: str,
    *,
    phase_assignments: Mapping[str, Tuple[RTPlanPhase, float]],
    single_phase: Optional[RTPlanPhase],
    plan_phases: Sequence[RTPlanPhase],
) -> int:
    phase_assignment = phase_assignments.get(normalized_name)
    if phase_assignment is not None:
        return int(max(phase_assignment[0].fractions_planned, 0))
    if single_phase is not None and source_key == single_phase.dose_path:
        return int(max(single_phase.fractions_planned, 0))
    available_fraction_counts = [
        int(max(phase.fractions_planned, 0))
        for phase in _get_available_phases(plan_phases)
        if phase.fractions_planned > 0
    ]
    if len(set(available_fraction_counts)) == 1 and available_fraction_counts:
        return available_fraction_counts[0]
    return 0


def get_primary_target_context(
    structure: StructureSliceContours,
    *,
    phase_assignments: Mapping[str, Tuple[RTPlanPhase, float]],
    single_phase: Optional[RTPlanPhase],
    parse_ptv_rx_gy_from_name: Callable[[str], Optional[float]],
    get_stereotactic_threshold_gy: Callable[[str, str], Optional[float]],
    compute_structure_target_metric_values: Callable[
        [StructureSliceContours, float, str],
        Tuple[Optional[float], Optional[float], Optional[float]],
    ],
    has_sampled_dose_volume_ct: bool,
) -> Tuple[float, str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    normalized_name = normalize_structure_name(structure.name)
    total_rx_gy = parse_ptv_rx_gy_from_name(structure.name)
    natural_reference_gy = total_rx_gy if total_rx_gy is not None else 0.0
    source_key = "combined"
    phase_assignment = phase_assignments.get(normalized_name)
    if phase_assignment is not None:
        phase, phase_rx_gy = phase_assignment
        natural_reference_gy = phase_rx_gy
        source_key = phase.dose_path
    reference_override_gy = get_stereotactic_threshold_gy(normalized_name, structure.name)
    rx_reference_gy = reference_override_gy if reference_override_gy is not None else natural_reference_gy

    metric_values: Tuple[Optional[float], Optional[float], Optional[float]] = (None, None, None)
    if rx_reference_gy > 0.0 and source_key != "combined":
        metric_values = compute_structure_target_metric_values(
            structure,
            rx_reference_gy,
            source_key,
        )
        return rx_reference_gy, source_key, metric_values

    if rx_reference_gy > 0.0 and has_sampled_dose_volume_ct:
        metric_values = compute_structure_target_metric_values(
            structure,
            rx_reference_gy,
            source_key,
        )
        if metric_values[0] is None and single_phase is not None:
            source_key = single_phase.dose_path
            metric_values = compute_structure_target_metric_values(
                structure,
                rx_reference_gy,
                source_key,
            )

    return rx_reference_gy, source_key, metric_values


def get_phase_target_assignments(
    plan_phases: Sequence[RTPlanPhase],
    sorted_ptv_structures: Sequence[StructureSliceContours],
    *,
    parse_ptv_rx_gy_from_name: Callable[[str], Optional[float]],
) -> Dict[str, Tuple[RTPlanPhase, float]]:
    available_phases = _get_available_phases(plan_phases)
    if not available_phases:
        return {}

    ptv_candidates: List[Tuple[StructureSliceContours, str, Optional[float]]] = []
    for structure in sorted_ptv_structures:
        normalized_name = normalize_structure_name(structure.name)
        total_rx_gy = parse_ptv_rx_gy_from_name(structure.name)
        ptv_candidates.append((structure, normalized_name, total_rx_gy))
    if not ptv_candidates:
        return {}

    if len(available_phases) == 1:
        phase = available_phases[0]
        exact_name_match = next(
            (
                normalized_name
                for _structure, normalized_name, _total_rx_gy in ptv_candidates
                if phase.target_structure_name == normalized_name
            ),
            None,
        )
        if exact_name_match is not None:
            return {exact_name_match: (phase, phase.prescription_dose_gy)}

        numeric_candidates = [
            (_structure, normalized_name, total_rx_gy)
            for _structure, normalized_name, total_rx_gy in ptv_candidates
            if total_rx_gy is not None
        ]
        if not numeric_candidates:
            return {}
        _best_structure, best_normalized_name, best_rx_gy = min(
            numeric_candidates,
            key=lambda item: (
                abs(float(item[2]) - phase.prescription_dose_gy),
                item[1],
            ),
        )
        if abs(float(best_rx_gy) - phase.prescription_dose_gy) <= 0.5:
            return {best_normalized_name: (phase, phase.prescription_dose_gy)}
        return {}

    assignments: Dict[str, Tuple[RTPlanPhase, float]] = {}
    cumulative_rx_gy = 0.0
    used_plan_uids: set[str] = set()

    for _structure, normalized_name, total_rx_gy in ptv_candidates:
        if total_rx_gy is None:
            continue
        incremental_rx_gy = max(total_rx_gy - cumulative_rx_gy, 0.0)
        if incremental_rx_gy <= 0.0:
            cumulative_rx_gy = total_rx_gy
            continue

        remaining_phases = [
            phase for phase in available_phases if phase.sop_instance_uid not in used_plan_uids
        ]
        if not remaining_phases:
            break

        best_phase = min(
            remaining_phases,
            key=lambda phase: (
                abs(phase.prescription_dose_gy - incremental_rx_gy),
                0 if phase.target_structure_name == normalized_name else 1,
                phase.plan_label,
            ),
        )
        if (
            abs(best_phase.prescription_dose_gy - incremental_rx_gy) > 0.5
            and best_phase.target_structure_name != normalized_name
        ):
            cumulative_rx_gy = total_rx_gy
            continue

        assignments[normalized_name] = (best_phase, incremental_rx_gy)
        used_plan_uids.add(best_phase.sop_instance_uid)
        cumulative_rx_gy = total_rx_gy

    return assignments


def get_default_stereotactic_dose_text(
    structure_name: str,
    *,
    plan_phases: Sequence[RTPlanPhase],
    constraints_sheet_name: str,
    phase_assignments: Mapping[str, Tuple[RTPlanPhase, float]],
    infer_srs_target_rx_gy: Callable[[str], Optional[float]],
) -> str:
    normalized_name = normalize_structure_name(structure_name)
    available_phases = _get_available_phases(plan_phases)
    fraction_counts = {
        int(max(phase.fractions_planned, 0))
        for phase in available_phases
        if phase.fractions_planned > 0
    }
    stereotactic_enabled = stereotactic_summary_enabled(constraints_sheet_name)
    is_single_fraction_srs = (
        stereotactic_enabled
        and bool(available_phases)
        and fraction_counts == {1}
    )
    is_multifraction_fsrt = (
        stereotactic_enabled
        and bool(available_phases)
        and len(fraction_counts) == 1
        and next(iter(fraction_counts), 0) > 1
    )

    phase_assignment = phase_assignments.get(normalized_name)
    if phase_assignment is not None:
        _phase, phase_rx_gy = phase_assignment
        if phase_rx_gy > 0.0:
            return f"{phase_rx_gy:.2f}"

    digits = "".join(ch for ch in normalized_name if ch.isdigit())
    raw_value = int(digits) if digits else None
    if raw_value is not None and raw_value >= 100:
        return f"{raw_value / 100.0:.2f}"
    is_ambiguous_stereotactic_ptv = (
        normalized_name.startswith("PTV")
        and stereotactic_enabled
        and (raw_value is None or raw_value < 100)
    )
    if is_ambiguous_stereotactic_ptv:
        if is_single_fraction_srs:
            inferred_rx_gy = infer_srs_target_rx_gy(normalized_name)
            if inferred_rx_gy is not None:
                return f"{inferred_rx_gy:.2f}"
        if is_multifraction_fsrt:
            if phase_assignment is not None:
                _phase, phase_rx_gy = phase_assignment
                if phase_rx_gy > 0.0:
                    return f"{phase_rx_gy:.2f}"
            if len(available_phases) == 1:
                return f"{available_phases[0].prescription_dose_gy:.2f}"
            return f"{sum(phase.prescription_dose_gy for phase in available_phases):.2f}"

    if digits:
        raw_value = int(digits)
        if normalized_name.startswith("PTV") and is_multifraction_fsrt and raw_value < 100:
            if len(available_phases) == 1:
                return f"{available_phases[0].prescription_dose_gy:.2f}"
            return f"{sum(phase.prescription_dose_gy for phase in available_phases):.2f}"
    elif normalized_name.startswith("PTV") and is_multifraction_fsrt:
        if len(available_phases) == 1:
            return f"{available_phases[0].prescription_dose_gy:.2f}"
        return f"{sum(phase.prescription_dose_gy for phase in available_phases):.2f}"

    if stereotactic_enabled:
        inferred_rx_gy = infer_srs_target_rx_gy(normalized_name)
        if inferred_rx_gy is not None:
            return f"{inferred_rx_gy:.2f}"

    available_phase_rx = [phase.prescription_dose_gy for phase in available_phases]
    if len(available_phase_rx) == 1:
        return f"{available_phase_rx[0]:.2f}"
    if available_phase_rx:
        return f"{sum(available_phase_rx):.2f}"
    return ""


def _get_brain_metric_text(fractions_planned: int, brain_metric_cc: float) -> str:
    if fractions_planned == 1:
        return f"V12Gy (Brain-GTV) {brain_metric_cc:.1f} cc"
    if fractions_planned == 5:
        return f"V24Gy (Brain) {brain_metric_cc:.1f} cc"
    return ""


def compute_stereotactic_indices(
    structure: StructureSliceContours,
    threshold_gy: float,
    coverage_pct: float,
    source_key: str,
    fractions_planned: int,
    *,
    stereotactic_metrics_cache: Dict[Tuple[str, str, float, int], Tuple[float, float, float, float, float]],
    get_structure_geometry_volume_cc: Callable[[StructureSliceContours], float],
    get_stereotactic_volume_context: Callable[[StructureSliceContours, str, float], Optional[Mapping[str, object]]],
    compute_stereotactic_owned_volume_cc: Callable[[Mapping[str, object], float, Optional[np.ndarray]], float],
    get_brain_structure: Callable[[], Optional[StructureSliceContours]],
    get_structure_volume_mask: Callable[[StructureSliceContours], np.ndarray],
    get_nested_target_structures: Callable[[StructureSliceContours], Sequence[StructureSliceContours]],
) -> Tuple[str, str, str, str, str]:
    normalized_name = normalize_structure_name(structure.name)
    cache_key = (normalized_name, source_key, round(float(threshold_gy), 3), int(fractions_planned))
    cached = stereotactic_metrics_cache.get(cache_key)
    if cached is not None:
        volume_cc, ci_value, pci_value, gi_value, brain_metric_cc = cached
        return (
            f"Vol {volume_cc:.2f} cc",
            f"CI {ci_value:.2f}",
            f"PCI {pci_value:.2f}",
            f"GI {gi_value:.2f}",
            _get_brain_metric_text(fractions_planned, brain_metric_cc),
        )

    ptv_volume_cc = get_structure_geometry_volume_cc(structure)
    if ptv_volume_cc <= 0.0:
        return "", "", "", "", ""

    ownership_thresholds = [threshold_gy, threshold_gy * 0.5]
    if fractions_planned == 1:
        ownership_thresholds.append(12.0)
    elif fractions_planned == 5:
        ownership_thresholds.append(24.0)
    context = get_stereotactic_volume_context(
        structure,
        source_key,
        min(ownership_thresholds),
    )
    if context is None:
        return "", "", "", "", ""

    ci_isodose_volume_cc = compute_stereotactic_owned_volume_cc(
        context,
        threshold_gy,
        None,
    )
    gi_reference_volume_cc = ci_isodose_volume_cc
    gradient_isodose_volume_cc = compute_stereotactic_owned_volume_cc(
        context,
        threshold_gy * 0.5,
        None,
    )
    ci_value = ci_isodose_volume_cc / ptv_volume_cc
    coverage_decimal = float(np.clip(coverage_pct / 100.0, 0.0, 1.0))
    pci_value = float((coverage_decimal ** 2) / ci_value) if ci_value > 0.0 else 0.0
    gi_value = gradient_isodose_volume_cc / gi_reference_volume_cc if gi_reference_volume_cc > 0.0 else 0.0
    brain_metric_cc = 0.0
    brain_structure = get_brain_structure()
    if brain_structure is not None and fractions_planned in {1, 5}:
        extra_mask = get_structure_volume_mask(brain_structure)
        if fractions_planned == 1:
            nested_gtv_structures = [
                nested_structure
                for nested_structure in get_nested_target_structures(structure)
                if normalize_structure_name(nested_structure.name).startswith("GTV")
            ]
            if nested_gtv_structures:
                gtv_union_mask = np.zeros_like(extra_mask, dtype=bool)
                for nested_gtv_structure in nested_gtv_structures:
                    gtv_union_mask |= get_structure_volume_mask(nested_gtv_structure)
                extra_mask = extra_mask & ~gtv_union_mask
            brain_metric_cc = compute_stereotactic_owned_volume_cc(
                context,
                12.0,
                extra_mask,
            )
        elif fractions_planned == 5:
            brain_metric_cc = compute_stereotactic_owned_volume_cc(
                context,
                24.0,
                extra_mask,
            )

    stereotactic_metrics_cache[cache_key] = (
        ptv_volume_cc,
        ci_value,
        pci_value,
        gi_value,
        brain_metric_cc,
    )
    return (
        f"Vol {ptv_volume_cc:.2f} cc",
        f"CI {ci_value:.2f}",
        f"PCI {pci_value:.2f}",
        f"GI {gi_value:.2f}",
        _get_brain_metric_text(fractions_planned, brain_metric_cc),
    )


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


def build_target_table_rows(
    *,
    rtstruct: Optional[RTStructData],
    has_ct: bool,
    plan_phases: Sequence[RTPlanPhase],
    sorted_ptv_structures: Sequence[StructureSliceContours],
    phase_assignments: Mapping[str, Tuple[RTPlanPhase, float]],
    stereotactic_summary_enabled: bool,
    parse_ptv_rx_gy_from_name: Callable[[str], Optional[float]],
    get_stereotactic_threshold_gy: Callable[[str, str], Optional[float]],
    get_primary_target_context: Callable[
        [StructureSliceContours, Mapping[str, Tuple[RTPlanPhase, float]], Optional[RTPlanPhase]],
        Tuple[float, str, Tuple[Optional[float], Optional[float], Optional[float]]],
    ],
    get_target_fraction_count: Callable[
        [str, str, Mapping[str, Tuple[RTPlanPhase, float]], Optional[RTPlanPhase]],
        int,
    ],
    compute_stereotactic_indices: Callable[
        [StructureSliceContours, float, float, str, int],
        Tuple[str, str, str, str, str],
    ],
    get_nested_target_structures: Callable[[StructureSliceContours], Sequence[StructureSliceContours]],
    compute_structure_target_metric_values: Callable[
        [StructureSliceContours, float, str],
        Tuple[Optional[float], Optional[float], Optional[float]],
    ],
    compute_structure_target_metrics: Callable[
        [StructureSliceContours, float, str],
        Tuple[str, str, str],
    ],
    format_target_dose_text: Callable[[float, float], str],
) -> List[Dict[str, object]]:
    if rtstruct is None or not has_ct:
        return []

    single_phase: Optional[RTPlanPhase] = None
    available_phases = _get_available_phases(plan_phases)
    if len(available_phases) == 1:
        single_phase = available_phases[0]

    rows: List[Dict[str, object]] = []
    for structure in sorted_ptv_structures:
        normalized_name = normalize_structure_name(structure.name)
        initial_reference_gy = get_stereotactic_threshold_gy(normalized_name, structure.name)

        minimum_dose_text = ""
        maximum_dose_text = ""
        coverage_text = (
            f"@ {initial_reference_gy:.2f} Gy"
            if initial_reference_gy is not None and initial_reference_gy > 0.0
            else ""
        )
        stereotactic_summary_text = ""
        (
            rx_reference_gy,
            source_key,
            primary_metric_values,
        ) = get_primary_target_context(
            structure,
            phase_assignments,
            single_phase,
        )
        fractions_planned = get_target_fraction_count(
            normalized_name,
            source_key,
            phase_assignments,
            single_phase,
        )
        min_dose_gy, max_dose_gy, coverage_pct = primary_metric_values
        if min_dose_gy is not None and max_dose_gy is not None and coverage_pct is not None:
            minimum_dose_text = format_target_dose_text(min_dose_gy, rx_reference_gy)
            maximum_dose_text = format_target_dose_text(max_dose_gy, rx_reference_gy)
            coverage_text = f"{coverage_pct:.1f}% @ {rx_reference_gy:.2f} Gy"

        rows.append(
            {
                "structure_name": structure.name,
                "normalized_name": normalized_name,
                "parent_structure_name": None,
                "parent_normalized_name": None,
                "display_name": structure.name,
                "reference_dose_text": f"{rx_reference_gy:.2f}" if rx_reference_gy > 0.0 else "",
                "coverage_text": coverage_text,
                "minimum_dose_text": minimum_dose_text,
                "maximum_dose_text": maximum_dose_text,
                "notes_text": stereotactic_summary_text,
                "is_primary_ptv": True,
                "color_rgb": structure.color_rgb,
            }
        )

        if stereotactic_summary_enabled:
            stereotactic_volume_text = ""
            stereotactic_ci_text = ""
            stereotactic_pci_text = ""
            stereotactic_gi_text = ""
            stereotactic_brain_metric_text = ""
            stereotactic_hi_text = ""
            if coverage_pct is not None and rx_reference_gy > 0.0:
                (
                    stereotactic_volume_text,
                    stereotactic_ci_text,
                    stereotactic_pci_text,
                    stereotactic_gi_text,
                    stereotactic_brain_metric_text,
                ) = compute_stereotactic_indices(
                    structure,
                    rx_reference_gy,
                    coverage_pct,
                    source_key,
                    fractions_planned,
                )
            if max_dose_gy is not None and rx_reference_gy > 0.0:
                stereotactic_hi_text = f"HI {max_dose_gy / rx_reference_gy:.2f}"
            stereotactic_parts = [
                text
                for text in (
                    stereotactic_volume_text,
                    stereotactic_ci_text,
                    stereotactic_pci_text,
                    stereotactic_gi_text,
                    stereotactic_hi_text,
                    stereotactic_brain_metric_text,
                )
                if text
            ]
            if stereotactic_parts:
                rows[-1]["notes_text"] = "    ".join(stereotactic_parts)

        for nested_structure in get_nested_target_structures(structure):
            nested_normalized_name = normalize_structure_name(nested_structure.name)
            nested_minimum_dose_text = ""
            nested_maximum_dose_text = ""
            nested_coverage_text = f"@ {rx_reference_gy:.2f} Gy"
            if rx_reference_gy > 0.0:
                nested_metric_source_key = source_key
                nested_metric_values = compute_structure_target_metric_values(
                    nested_structure,
                    rx_reference_gy,
                    nested_metric_source_key,
                )
                if nested_metric_values[0] is None and nested_metric_source_key != "combined":
                    nested_metric_source_key = "combined"
                nested_minimum_dose_text, nested_maximum_dose_text, nested_coverage_text = (
                    compute_structure_target_metrics(
                        nested_structure,
                        rx_reference_gy,
                        nested_metric_source_key,
                    )
                )
            rows.append(
                {
                    "structure_name": nested_structure.name,
                    "normalized_name": nested_normalized_name,
                    "parent_structure_name": structure.name,
                    "parent_normalized_name": normalized_name,
                    "display_name": f"    {nested_structure.name}",
                    "reference_dose_text": f"{rx_reference_gy:.2f}" if rx_reference_gy > 0.0 else "",
                    "coverage_text": nested_coverage_text,
                    "minimum_dose_text": nested_minimum_dose_text,
                    "maximum_dose_text": nested_maximum_dose_text,
                    "notes_text": "",
                    "is_primary_ptv": False,
                    "color_rgb": nested_structure.color_rgb,
                }
            )

    return rows


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
