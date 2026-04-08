from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from peer_helpers import normalize_structure_name
from peer_models import (
    CTVolume,
    DVHCurve,
    RTStructData,
    StructureGoal,
    StructureGoalEvaluation,
    StructureSliceContours,
)
from peer_targets import target_table_rows_require_recompute
from peer_viewer_support import (
    build_file_fingerprint,
    build_file_fingerprints,
    file_fingerprint_list_matches,
    file_fingerprint_matches,
)


def callable_signature_hash(func: Callable[..., object]) -> str:
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = repr(func)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def get_dvh_cache_path(current_patient_folder: Optional[str]) -> Optional[Path]:
    if current_patient_folder is None:
        return None
    return Path(current_patient_folder) / "peer_dvh_constraints.json"


def get_derived_array_cache_path(base_path: Optional[Path]) -> Optional[Path]:
    if base_path is None:
        return None
    return base_path.with_name(f"{base_path.stem}_arrays.npz")


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path_str = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=path.suffix, dir=path.parent)
    temp_path = Path(temp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        temp_path.replace(path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def get_ct_geometry_signature(ct: Optional[CTVolume]) -> Optional[str]:
    if ct is None:
        return None
    hasher = hashlib.sha256()
    hasher.update(np.asarray(ct.volume_hu.shape, dtype=np.int32).tobytes())
    hasher.update(np.asarray(ct.spacing_xyz_mm, dtype=np.float32).tobytes())
    hasher.update(np.asarray(ct.z_positions_mm, dtype=np.float32).tobytes())
    return hasher.hexdigest()[:16]


def get_derived_array_cache_signature(
    *,
    sample_dose_to_ct_slice_func: Callable[..., object],
    build_structure_slice_mask_func: Callable[..., object],
    get_target_structure_slice_masks_func: Callable[..., object],
    get_ptv_union_slice_masks_func: Callable[..., object],
    load_rtstruct_func: Callable[..., object],
) -> Dict[str, str]:
    return {
        "sample_dose_to_ct_slice": callable_signature_hash(sample_dose_to_ct_slice_func),
        "build_structure_slice_mask": callable_signature_hash(build_structure_slice_mask_func),
        "get_target_structure_slice_masks": callable_signature_hash(get_target_structure_slice_masks_func),
        "get_ptv_union_slice_masks": callable_signature_hash(get_ptv_union_slice_masks_func),
        "load_rtstruct": callable_signature_hash(load_rtstruct_func),
    }


def get_derived_cache_structures(
    rtstruct: Optional[RTStructData],
    additional_target_subvolume_names: set[str],
) -> List[StructureSliceContours]:
    if rtstruct is None:
        return []

    structures: List[StructureSliceContours] = []
    seen_names: set[str] = set()
    for structure in rtstruct.structures:
        normalized_name = normalize_structure_name(structure.name)
        include = (
            normalized_name.startswith(("PTV", "GTV", "CTV"))
            or normalized_name in additional_target_subvolume_names
            or normalized_name == "BRAIN"
            or (normalized_name.startswith("BRAIN") and "BRAINSTEM" not in normalized_name)
        )
        if not include or normalized_name in seen_names:
            continue
        structures.append(structure)
        seen_names.add(normalized_name)
    return structures


def default_is_base_listable_structure_name(normalized_name: str) -> bool:
    excluded_fragments = ("COUCH", "RAIL", "BB")
    return not normalized_name.startswith("Z") and not any(
        fragment in normalized_name for fragment in excluded_fragments
    )


@dataclass(slots=True)
class DerivedArrayCacheData:
    rtstruct: Optional[RTStructData]
    sampled_dose_volume_ct: Optional[np.ndarray]
    ptv_union_volume_mask: Optional[np.ndarray]
    structure_volume_masks: Dict[str, np.ndarray]
    structure_geometry_volumes_cc: Dict[str, float]


@dataclass(slots=True)
class ReviewCacheFileData:
    payload: Dict[str, object]
    cache_version: int


@dataclass(slots=True)
class PreparedReviewCacheState:
    selected_constraint_sheet: Optional[str]
    custom_constraints: Dict[str, List[StructureGoal]]
    stereotactic_target_doses: Dict[str, str]
    isodose_level_texts: Optional[List[str]]
    isodose_colors: Optional[List[Tuple[int, int, int]]]
    hidden_structure_names: set[str]
    additional_target_subvolume_names: set[str]
    constraint_notes: Dict[str, str]
    target_notes_payload: Dict[str, str]
    target_table_rows: List[Dict[str, object]]
    cached_target_table_rows: Optional[List[Dict[str, object]]]
    goal_evaluations: Dict[str, List[StructureGoalEvaluation]]
    curves: List[DVHCurve]
    max_tissue_dose_gy: Optional[float]
    max_tissue_index_zyx: Optional[Tuple[int, int, int]]
    saved_selected_names: Optional[List[str]]


def _serialize_rtstruct_geometry(
    rtstruct: Optional[RTStructData],
    arrays: Dict[str, np.ndarray],
) -> Dict[str, object]:
    if rtstruct is None:
        return {}

    structure_entries: List[Dict[str, object]] = []
    for structure_index, structure in enumerate(rtstruct.structures):
        contour_arrays: List[np.ndarray] = []
        contour_slice_indices: List[int] = []
        for slice_index in sorted(structure.points_rc_by_slice.keys()):
            for contour_rc in structure.points_rc_by_slice.get(slice_index, []):
                contour_array = np.asarray(contour_rc, dtype=np.float32)
                if contour_array.ndim != 2 or contour_array.shape[1] != 2:
                    continue
                contour_arrays.append(contour_array)
                contour_slice_indices.append(int(slice_index))

        points_key = f"rtstruct_points_{structure_index:03d}"
        offsets_key = f"rtstruct_offsets_{structure_index:03d}"
        slices_key = f"rtstruct_slices_{structure_index:03d}"

        if contour_arrays:
            stacked_points = np.concatenate(contour_arrays, axis=0).astype(np.float32, copy=False)
            contour_lengths = np.asarray([contour.shape[0] for contour in contour_arrays], dtype=np.int32)
            contour_offsets = np.concatenate(
                (
                    np.asarray([0], dtype=np.int32),
                    np.cumsum(contour_lengths, dtype=np.int32),
                )
            )
        else:
            stacked_points = np.zeros((0, 2), dtype=np.float32)
            contour_offsets = np.zeros((1,), dtype=np.int32)

        arrays[points_key] = stacked_points
        arrays[offsets_key] = contour_offsets
        arrays[slices_key] = np.asarray(contour_slice_indices, dtype=np.int32)
        structure_entries.append(
            {
                "name": structure.name,
                "color_rgb": [int(component) for component in structure.color_rgb[:3]],
                "points_key": points_key,
                "offsets_key": offsets_key,
                "slices_key": slices_key,
            }
        )

    return {
        "frame_of_reference_uid": rtstruct.frame_of_reference_uid,
        "structures": structure_entries,
    }


def _deserialize_rtstruct_geometry(
    payload: Mapping[str, object],
    arrays: Any,
) -> Optional[RTStructData]:
    structures_payload = payload.get("structures")
    if not isinstance(structures_payload, list):
        return None

    structures: List[StructureSliceContours] = []
    for structure_entry in structures_payload:
        if not isinstance(structure_entry, dict):
            return None
        name = str(structure_entry.get("name", ""))
        if not name:
            return None
        color_payload = structure_entry.get("color_rgb", [255, 255, 255])
        if not isinstance(color_payload, list) or len(color_payload) < 3:
            return None
        try:
            color_rgb = tuple(int(component) for component in color_payload[:3])
        except (TypeError, ValueError):
            return None

        points_key = str(structure_entry.get("points_key", ""))
        offsets_key = str(structure_entry.get("offsets_key", ""))
        slices_key = str(structure_entry.get("slices_key", ""))
        if not points_key or not offsets_key or not slices_key:
            return None

        try:
            contour_points = np.asarray(arrays[points_key], dtype=np.float32)
            contour_offsets = np.asarray(arrays[offsets_key], dtype=np.int32)
            contour_slices = np.asarray(arrays[slices_key], dtype=np.int32)
        except KeyError:
            return None

        if contour_points.ndim != 2 or contour_points.shape[1] != 2:
            return None
        if contour_offsets.ndim != 1 or contour_offsets.size != contour_slices.size + 1:
            return None
        if contour_offsets.size == 0 or int(contour_offsets[0]) != 0 or int(contour_offsets[-1]) != contour_points.shape[0]:
            return None

        points_rc_by_slice: Dict[int, List[np.ndarray]] = {}
        for contour_index, slice_index in enumerate(contour_slices.tolist()):
            start = int(contour_offsets[contour_index])
            end = int(contour_offsets[contour_index + 1])
            if start < 0 or end < start or end > contour_points.shape[0]:
                return None
            contour_rc = np.asarray(contour_points[start:end], dtype=np.float32).copy()
            points_rc_by_slice.setdefault(int(slice_index), []).append(contour_rc)

        structures.append(
            StructureSliceContours(
                name=name,
                color_rgb=color_rgb,  # type: ignore[arg-type]
                points_rc_by_slice=points_rc_by_slice,
            )
        )

    return RTStructData(
        structures=structures,
        frame_of_reference_uid=str(payload.get("frame_of_reference_uid", "")),
    )


def json_safe_metadata_value(value: object) -> object:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def serialize_dvh_curve(curve: DVHCurve) -> Dict[str, object]:
    return {
        "name": curve.name,
        "color_rgb": list(curve.color_rgb),
        "dose_bins_gy": [float(value) for value in curve.dose_bins_gy],
        "volume_pct": [float(value) for value in curve.volume_pct],
        "voxel_count": int(curve.voxel_count),
        "volume_cc": float(curve.volume_cc),
        "mean_dose_gy": float(curve.mean_dose_gy),
        "max_dose_gy": float(curve.max_dose_gy),
        "min_dose_gy": float(curve.min_dose_gy),
        "volume_cc_axis": [float(value) for value in curve.volume_cc_axis],
        "oversampling_factor": float(curve.oversampling_factor),
        "used_fractional_labelmap": bool(curve.used_fractional_labelmap),
        "metadata": {
            str(key): json_safe_metadata_value(value)
            for key, value in curve.metadata.items()
        },
    }


def deserialize_dvh_curve(payload: Mapping[str, object]) -> DVHCurve:
    return DVHCurve(
        name=str(payload.get("name", "")),
        color_rgb=tuple(int(value) for value in payload.get("color_rgb", [255, 255, 255])),
        dose_bins_gy=np.asarray(payload.get("dose_bins_gy", []), dtype=np.float32),
        volume_pct=np.asarray(payload.get("volume_pct", []), dtype=np.float32),
        voxel_count=int(payload.get("voxel_count", 0)),
        volume_cc=float(payload.get("volume_cc", 0.0)),
        mean_dose_gy=float(payload.get("mean_dose_gy", 0.0)),
        max_dose_gy=float(payload.get("max_dose_gy", 0.0)),
        min_dose_gy=float(payload.get("min_dose_gy", 0.0)),
        volume_cc_axis=np.asarray(payload.get("volume_cc_axis", []), dtype=np.float32),
        oversampling_factor=float(payload.get("oversampling_factor", 1.0)),
        used_fractional_labelmap=bool(payload.get("used_fractional_labelmap", False)),
        metadata=dict(payload.get("metadata", {})),
    )


def serialize_goal_evaluations(
    structure_goal_evaluations: Mapping[str, Sequence[StructureGoalEvaluation]],
) -> Dict[str, List[Dict[str, object]]]:
    serialized: Dict[str, List[Dict[str, object]]] = {}
    for structure_name, evaluations in structure_goal_evaluations.items():
        serialized[structure_name] = [
            {
                "metric": evaluation.metric,
                "comparator": evaluation.comparator,
                "goal_text": evaluation.goal_text,
                "actual_text": evaluation.actual_text,
                "passed": evaluation.passed,
                "status": evaluation.status,
            }
            for evaluation in evaluations
        ]
    return serialized


def deserialize_goal_evaluations(
    payload: object,
) -> Optional[Dict[str, List[StructureGoalEvaluation]]]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        return None

    evaluations_by_structure: Dict[str, List[StructureGoalEvaluation]] = {}
    for structure_name, evaluation_payloads in payload.items():
        if not isinstance(evaluation_payloads, list):
            return None
        evaluations: List[StructureGoalEvaluation] = []
        for evaluation_payload in evaluation_payloads:
            if not isinstance(evaluation_payload, dict):
                return None
            evaluations.append(
                StructureGoalEvaluation(
                    metric=str(evaluation_payload.get("metric", "")),
                    comparator=str(evaluation_payload.get("comparator", "")),
                    goal_text=str(evaluation_payload.get("goal_text", "")),
                    actual_text=str(evaluation_payload.get("actual_text", "")),
                    passed=evaluation_payload.get("passed"),
                    status=str(evaluation_payload.get("status", "")),
                )
            )
        evaluations_by_structure[normalize_structure_name(str(structure_name))] = evaluations
    return evaluations_by_structure


def serialize_structure_goals(
    goals_by_structure: Mapping[str, Sequence[StructureGoal]],
) -> Dict[str, List[Dict[str, str]]]:
    serialized: Dict[str, List[Dict[str, str]]] = {}
    for structure_name, goals in goals_by_structure.items():
        serialized[structure_name] = [
            {
                "structure_name": goal.structure_name,
                "metric": goal.metric,
                "comparator": goal.comparator,
                "value_text": goal.value_text,
            }
            for goal in goals
        ]
    return serialized


def deserialize_structure_goals(
    payload: object,
) -> Optional[Dict[str, List[StructureGoal]]]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        return None

    goals_by_structure: Dict[str, List[StructureGoal]] = {}
    for structure_name, goal_payloads in payload.items():
        if not isinstance(goal_payloads, list):
            return None
        goals: List[StructureGoal] = []
        for goal_payload in goal_payloads:
            if not isinstance(goal_payload, dict):
                return None
            goals.append(
                StructureGoal(
                    structure_name=str(goal_payload.get("structure_name", structure_name)),
                    metric=str(goal_payload.get("metric", "")),
                    comparator=str(goal_payload.get("comparator", "")),
                    value_text=str(goal_payload.get("value_text", "")),
                )
            )
        goals_by_structure[normalize_structure_name(str(structure_name))] = goals
    return goals_by_structure


def serialize_target_table_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    serialized_rows: List[Dict[str, object]] = []
    for row in rows:
        serialized_rows.append(
            {
                "structure_name": str(row.get("structure_name", "")),
                "normalized_name": str(row.get("normalized_name", "")),
                "parent_structure_name": row.get("parent_structure_name"),
                "parent_normalized_name": row.get("parent_normalized_name"),
                "display_name": str(row.get("display_name", "")),
                "reference_dose_text": str(row.get("reference_dose_text", "")),
                "coverage_text": str(row.get("coverage_text", "")),
                "minimum_dose_text": str(row.get("minimum_dose_text", "")),
                "maximum_dose_text": str(row.get("maximum_dose_text", "")),
                "notes_text": str(row.get("notes_text", "")),
                "is_primary_ptv": bool(row.get("is_primary_ptv", False)),
                "color_rgb": [int(value) for value in row.get("color_rgb", [255, 255, 255])],
            }
        )
    return serialized_rows


def deserialize_target_table_rows(payload: object) -> Optional[List[Dict[str, object]]]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        return None

    rows: List[Dict[str, object]] = []
    for row_payload in payload:
        if not isinstance(row_payload, dict):
            return None
        color_values = row_payload.get("color_rgb", [255, 255, 255])
        if not isinstance(color_values, list) or len(color_values) != 3:
            return None
        rows.append(
            {
                "structure_name": str(row_payload.get("structure_name", "")),
                "normalized_name": normalize_structure_name(str(row_payload.get("normalized_name", ""))),
                "parent_structure_name": (
                    None
                    if row_payload.get("parent_structure_name") in {None, ""}
                    else str(row_payload.get("parent_structure_name"))
                ),
                "parent_normalized_name": (
                    None
                    if row_payload.get("parent_normalized_name") in {None, ""}
                    else normalize_structure_name(str(row_payload.get("parent_normalized_name")))
                ),
                "display_name": str(row_payload.get("display_name", "")),
                "reference_dose_text": str(row_payload.get("reference_dose_text", "")),
                "coverage_text": str(row_payload.get("coverage_text", "")),
                "minimum_dose_text": str(row_payload.get("minimum_dose_text", "")),
                "maximum_dose_text": str(row_payload.get("maximum_dose_text", "")),
                "notes_text": str(row_payload.get("notes_text", "")),
                "is_primary_ptv": bool(row_payload.get("is_primary_ptv", False)),
                "color_rgb": tuple(int(value) for value in color_values),
            }
        )
    return rows


def build_review_cache_payload(
    *,
    patient_plan_lines: Optional[Sequence[str]],
    selected_constraint_set: str,
    constraints_file_name: Optional[str],
    constraints_sheet_name: Optional[str],
    rtstruct_file_name: Optional[str],
    constraints_fingerprint: Optional[Dict[str, object]],
    rtstruct_fingerprint: Optional[Dict[str, object]],
    rtdose_fingerprints: List[Dict[str, object]],
    rtplan_fingerprints: List[Dict[str, object]],
    derived_array_cache_file_name: Optional[str],
    derived_array_cache_signature: Dict[str, str],
    structure_names: Sequence[str],
    dvh_structure_names: Sequence[str],
    dvh_mode: str,
    dvh_method_signature: str,
    target_method_signature: Dict[str, object],
    curves: Sequence[DVHCurve],
    custom_constraints: Mapping[str, Sequence[StructureGoal]],
    goal_evaluations: Mapping[str, Sequence[StructureGoalEvaluation]],
    target_table_rows: Sequence[Mapping[str, object]],
    max_tissue_payload: Optional[Dict[str, object]],
    stereotactic_target_doses: Mapping[str, str],
    isodose_level_texts: Sequence[str],
    isodose_colors: Sequence[Sequence[int]],
    hidden_structure_names: Sequence[str],
    additional_target_subvolume_names: Sequence[str],
    constraint_notes: Mapping[str, str],
    target_notes: Mapping[str, str],
) -> Dict[str, object]:
    return {
        "version": 16,
        "saved_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "patient_plan_lines": list(patient_plan_lines or []),
        "selected_constraint_set": selected_constraint_set,
        "constraints_file": constraints_file_name,
        "constraints_sheet": constraints_sheet_name,
        "rtstruct_file": rtstruct_file_name,
        "constraints_fingerprint": constraints_fingerprint,
        "rtstruct_fingerprint": rtstruct_fingerprint,
        "rtdose_fingerprints": rtdose_fingerprints,
        "rtplan_fingerprints": rtplan_fingerprints,
        "derived_array_cache_file": derived_array_cache_file_name,
        "derived_array_cache_signature": derived_array_cache_signature,
        "structure_names": list(structure_names),
        "dvh_structure_names": list(dvh_structure_names),
        "dvh_mode": dvh_mode,
        "dvh_method_signature": dvh_method_signature,
        "target_method_signature": target_method_signature,
        "curves": [serialize_dvh_curve(curve) for curve in curves],
        "custom_constraints": serialize_structure_goals(custom_constraints),
        "goal_evaluations": serialize_goal_evaluations(goal_evaluations),
        "target_table_rows": serialize_target_table_rows(target_table_rows),
        "max_tissue": max_tissue_payload,
        "stereotactic_target_doses": dict(stereotactic_target_doses),
        "isodose_level_texts": [str(text) for text in isodose_level_texts],
        "isodose_colors": [
            [int(component) for component in color[:3]]
            for color in isodose_colors
            if len(color) >= 3
        ],
        "hidden_structure_names": list(hidden_structure_names),
        "additional_target_subvolume_names": list(additional_target_subvolume_names),
        "constraint_notes": dict(constraint_notes),
        "target_notes": dict(target_notes),
    }


def load_review_cache_file(path: Path) -> Optional[ReviewCacheFileData]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        cache_version = int(payload.get("version", 0))
    except (TypeError, ValueError):
        cache_version = 0
    return ReviewCacheFileData(payload=payload, cache_version=cache_version)


def prepare_review_cache_state(
    loaded_cache: ReviewCacheFileData,
    *,
    expected_structure_names: Sequence[str],
    available_constraint_sheet_names: Sequence[str],
    no_constraints_sheet_label: str,
    constraints_sheet_name: Optional[str],
    structure_filter_csv_path: Optional[str],
    rtstruct_path: Optional[str],
    rtdose_paths: Sequence[str],
    rtplan_paths: Sequence[str],
    dvh_mode: str,
    dvh_method_signature: str,
    target_method_signature: Dict[str, object],
    has_ct: bool,
    has_dose: bool,
    is_base_listable_structure_name: Callable[[str], bool],
) -> Optional[PreparedReviewCacheState]:
    payload = loaded_cache.payload
    cache_version = loaded_cache.cache_version

    expected_names = [normalize_structure_name(name) for name in expected_structure_names]
    expected_name_set = set(expected_names)
    saved_names = [normalize_structure_name(str(name)) for name in payload.get("structure_names", [])]
    saved_name_set = set(saved_names)
    saved_selected_names = [
        normalize_structure_name(str(name))
        for name in payload.get("dvh_structure_names", [])
    ]
    saved_selected_name_set = set(saved_selected_names)
    if not saved_selected_names:
        saved_selected_names = [
            normalize_structure_name(str(curve_payload.get("name", "")))
            for curve_payload in payload.get("curves", [])
            if isinstance(curve_payload, dict)
        ]
        saved_selected_name_set = set(saved_selected_names)

    if saved_names:
        if saved_name_set == expected_name_set:
            pass
        elif saved_name_set == saved_selected_name_set and saved_name_set.issubset(expected_name_set):
            pass
        else:
            return None

    if not saved_selected_name_set.issubset(expected_name_set):
        return None

    saved_constraint_set = payload.get("selected_constraint_set", payload.get("constraints_sheet"))
    if saved_constraint_set == no_constraints_sheet_label:
        saved_constraints_sheet = None
    elif saved_constraint_set in {None, ""}:
        saved_constraints_sheet = payload.get("constraints_sheet")
    else:
        saved_constraints_sheet = str(saved_constraint_set)

    if saved_constraints_sheet is not None and saved_constraints_sheet not in available_constraint_sheet_names:
        return None

    saved_csv_fingerprint = payload.get("constraints_fingerprint", payload.get("csv_fingerprint"))
    if saved_csv_fingerprint is not None and not file_fingerprint_matches(saved_csv_fingerprint, structure_filter_csv_path):
        return None

    saved_csv_name = payload.get("constraints_file", payload.get("csv_file"))
    current_csv_name = Path(structure_filter_csv_path).name if structure_filter_csv_path else None
    if saved_csv_fingerprint is None and saved_csv_name not in {None, current_csv_name}:
        return None

    saved_rtstruct_fingerprint = payload.get("rtstruct_fingerprint")
    if saved_rtstruct_fingerprint is not None and not file_fingerprint_matches(saved_rtstruct_fingerprint, rtstruct_path):
        return None

    saved_rtstruct_name = payload.get("rtstruct_file")
    current_rtstruct_name = Path(rtstruct_path).name if rtstruct_path else None
    if saved_rtstruct_fingerprint is None and saved_rtstruct_name not in {None, current_rtstruct_name}:
        return None

    saved_rtdose_fingerprints = payload.get("rtdose_fingerprints")
    if saved_rtdose_fingerprints is not None and not file_fingerprint_list_matches(saved_rtdose_fingerprints, list(rtdose_paths)):
        return None

    saved_rtplan_fingerprints = payload.get("rtplan_fingerprints")
    if saved_rtplan_fingerprints is not None and not file_fingerprint_list_matches(saved_rtplan_fingerprints, list(rtplan_paths)):
        return None

    saved_dvh_mode = payload.get("dvh_mode")
    if saved_dvh_mode not in {None, dvh_mode}:
        return None

    notes_payload = payload.get("constraint_notes", {})
    if not isinstance(notes_payload, dict):
        return None
    target_notes_payload = payload.get("target_notes", {})
    if target_notes_payload is None:
        target_notes_payload = {}
    if not isinstance(target_notes_payload, dict):
        return None

    custom_constraints = deserialize_structure_goals(payload.get("custom_constraints"))
    if custom_constraints is None:
        return None

    stereotactic_dose_payload = payload.get("stereotactic_target_doses", {})
    if stereotactic_dose_payload is None:
        stereotactic_dose_payload = {}
    if not isinstance(stereotactic_dose_payload, dict):
        return None

    isodose_level_texts_payload = payload.get("isodose_level_texts")
    saved_isodose_level_texts: Optional[List[str]] = None
    if isinstance(isodose_level_texts_payload, list):
        saved_isodose_level_texts = [str(value).strip() for value in isodose_level_texts_payload]

    isodose_colors_payload = payload.get("isodose_colors")
    saved_isodose_colors: Optional[List[Tuple[int, int, int]]] = None
    if isinstance(isodose_colors_payload, list):
        parsed_colors: List[Tuple[int, int, int]] = []
        for color_payload in isodose_colors_payload:
            if not isinstance(color_payload, list) or len(color_payload) != 3:
                parsed_colors = []
                break
            try:
                parsed_colors.append(tuple(int(value) for value in color_payload))
            except (TypeError, ValueError):
                parsed_colors = []
                break
        if parsed_colors:
            saved_isodose_colors = parsed_colors

    hidden_structure_payload = payload.get("hidden_structure_names", [])
    if hidden_structure_payload is None:
        hidden_structure_payload = []
    if not isinstance(hidden_structure_payload, list):
        return None

    additional_target_subvolume_payload = payload.get("additional_target_subvolume_names", [])
    if additional_target_subvolume_payload is None:
        additional_target_subvolume_payload = []
    if not isinstance(additional_target_subvolume_payload, list):
        return None

    target_table_rows = deserialize_target_table_rows(payload.get("target_table_rows"))
    if target_table_rows is None:
        return None

    max_tissue_payload = payload.get("max_tissue")
    if max_tissue_payload is not None and not isinstance(max_tissue_payload, dict):
        return None

    saved_dvh_signature = payload.get("dvh_method_signature")
    if saved_dvh_signature is not None and saved_dvh_signature != dvh_method_signature:
        return None

    saved_target_signature = payload.get("target_method_signature")
    if saved_target_signature is not None and saved_target_signature != target_method_signature:
        return None

    goal_evaluations = deserialize_goal_evaluations(payload.get("goal_evaluations"))
    if goal_evaluations is None:
        return None

    try:
        curves = [deserialize_dvh_curve(curve_payload) for curve_payload in payload.get("curves", [])]
    except (TypeError, ValueError):
        return None

    max_tissue_dose_gy: Optional[float] = None
    max_tissue_index_zyx: Optional[Tuple[int, int, int]] = None
    if isinstance(max_tissue_payload, dict):
        dose_value = max_tissue_payload.get("dose_gy")
        try:
            max_tissue_dose_gy = float(dose_value) if dose_value is not None else None
        except (TypeError, ValueError):
            max_tissue_dose_gy = None
        index_payload = max_tissue_payload.get("index_zyx")
        if isinstance(index_payload, list) and len(index_payload) == 3:
            try:
                max_tissue_index_zyx = tuple(int(value) for value in index_payload)
            except (TypeError, ValueError):
                max_tissue_index_zyx = None

    target_signature_matches = saved_target_signature is not None or cache_version >= 11
    use_cached_target_rows = target_signature_matches and not target_table_rows_require_recompute(
        target_table_rows,
        has_ct=has_ct,
        has_dose=has_dose,
        stereotactic_summary_enabled=normalize_structure_name(constraints_sheet_name or "") == "SRS FSRT",
    )

    return PreparedReviewCacheState(
        selected_constraint_sheet=saved_constraints_sheet,
        custom_constraints=custom_constraints,
        stereotactic_target_doses={
            normalize_structure_name(str(name)): str(value).strip()
            for name, value in stereotactic_dose_payload.items()
            if normalize_structure_name(str(name))
        },
        isodose_level_texts=saved_isodose_level_texts,
        isodose_colors=saved_isodose_colors,
        hidden_structure_names={
            normalize_structure_name(str(name))
            for name in hidden_structure_payload
            if is_base_listable_structure_name(normalize_structure_name(str(name)))
        },
        additional_target_subvolume_names={
            normalize_structure_name(str(name))
            for name in additional_target_subvolume_payload
            if is_base_listable_structure_name(normalize_structure_name(str(name)))
            and not normalize_structure_name(str(name)).startswith("PTV")
        },
        constraint_notes={str(key): str(value) for key, value in notes_payload.items() if str(value).strip()},
        target_notes_payload={str(key): str(value) for key, value in target_notes_payload.items() if str(value).strip()},
        target_table_rows=target_table_rows,
        cached_target_table_rows=target_table_rows if use_cached_target_rows else None,
        goal_evaluations=goal_evaluations or {},
        curves=curves,
        max_tissue_dose_gy=max_tissue_dose_gy,
        max_tissue_index_zyx=max_tissue_index_zyx,
        saved_selected_names=list(saved_selected_names) if saved_selected_names else None,
    )


def save_derived_array_cache(
    path: Path,
    *,
    ct: CTVolume,
    rtstruct: Optional[RTStructData],
    rtstruct_path: Optional[str],
    rtdose_paths: Sequence[str],
    array_cache_signature: Dict[str, str],
    sampled_dose_volume_ct: Optional[np.ndarray],
    ptv_union_volume_mask: Optional[np.ndarray],
    structure_order: Sequence[str],
    structure_volume_masks: Dict[str, np.ndarray],
    structure_geometry_volumes_cc: Dict[str, float],
) -> None:
    metadata: Dict[str, object] = {
        "version": 2,
        "saved_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "ct_geometry_signature": get_ct_geometry_signature(ct),
        "rtstruct_fingerprint": build_file_fingerprint(rtstruct_path),
        "rtdose_fingerprints": build_file_fingerprints(list(rtdose_paths)),
        "array_cache_signature": array_cache_signature,
        "structures": [],
        "rtstruct_geometry": {},
    }

    arrays: Dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata), dtype=np.str_),
    }
    metadata["rtstruct_geometry"] = _serialize_rtstruct_geometry(rtstruct, arrays)
    if sampled_dose_volume_ct is not None:
        arrays["sampled_dose_volume_ct"] = np.asarray(sampled_dose_volume_ct, dtype=np.float32)
    if ptv_union_volume_mask is not None:
        arrays["ptv_union_volume_mask"] = np.asarray(ptv_union_volume_mask, dtype=np.uint8)

    structure_entries: List[Dict[str, object]] = []
    for index, normalized_name in enumerate(structure_order):
        cached_mask = structure_volume_masks.get(normalized_name)
        if cached_mask is None:
            continue
        mask_key = f"structure_mask_{index:03d}"
        arrays[mask_key] = np.asarray(cached_mask, dtype=np.uint8)
        structure_entry: Dict[str, object] = {
            "name": normalized_name,
            "mask_key": mask_key,
        }
        geometry_volume_cc = structure_geometry_volumes_cc.get(normalized_name)
        if geometry_volume_cc is not None and geometry_volume_cc > 0.0:
            structure_entry["geometry_volume_cc"] = float(geometry_volume_cc)
        structure_entries.append(structure_entry)

    metadata["structures"] = structure_entries
    arrays["metadata_json"] = np.asarray(json.dumps(metadata), dtype=np.str_)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path_str = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_path_str)
    try:
        np.savez_compressed(temp_path, **arrays)
        temp_path.replace(path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_derived_array_cache(
    path: Path,
    *,
    ct: CTVolume,
    rtstruct_path: Optional[str],
    rtdose_paths: Sequence[str],
    array_cache_signature: Dict[str, str],
) -> Optional[DerivedArrayCacheData]:
    try:
        payload = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None

    try:
        metadata_json = payload["metadata_json"].item()
        metadata = json.loads(str(metadata_json))
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        payload.close()
        return None

    if not isinstance(metadata, dict):
        payload.close()
        return None
    cache_version = metadata.get("version")
    if cache_version not in {1, 2}:
        payload.close()
        return None
    if metadata.get("ct_geometry_signature") != get_ct_geometry_signature(ct):
        payload.close()
        return None
    if metadata.get("array_cache_signature") != array_cache_signature:
        payload.close()
        return None
    if not file_fingerprint_matches(metadata.get("rtstruct_fingerprint"), rtstruct_path):
        payload.close()
        return None
    if not file_fingerprint_list_matches(metadata.get("rtdose_fingerprints"), list(rtdose_paths)):
        payload.close()
        return None

    sampled_dose_volume_ct: Optional[np.ndarray] = None
    if "sampled_dose_volume_ct" in payload.files:
        sampled_dose_volume_ct = np.asarray(payload["sampled_dose_volume_ct"], dtype=np.float32)
        if sampled_dose_volume_ct.shape != ct.volume_hu.shape:
            payload.close()
            return None

    ptv_union_volume_mask: Optional[np.ndarray] = None
    if "ptv_union_volume_mask" in payload.files:
        ptv_union_volume_mask = np.asarray(payload["ptv_union_volume_mask"], dtype=bool)
        if ptv_union_volume_mask.shape != ct.volume_hu.shape:
            payload.close()
            return None

    structures_payload = metadata.get("structures", [])
    if not isinstance(structures_payload, list):
        payload.close()
        return None

    structure_volume_masks: Dict[str, np.ndarray] = {}
    structure_geometry_volumes_cc: Dict[str, float] = {}
    for structure_entry in structures_payload:
        if not isinstance(structure_entry, dict):
            payload.close()
            return None
        normalized_name = normalize_structure_name(str(structure_entry.get("name", "")))
        mask_key = str(structure_entry.get("mask_key", ""))
        if not normalized_name or not mask_key:
            payload.close()
            return None
        try:
            cached_mask = np.asarray(payload[mask_key], dtype=bool)
        except KeyError:
            payload.close()
            return None
        if cached_mask.shape != ct.volume_hu.shape:
            payload.close()
            return None
        structure_volume_masks[normalized_name] = cached_mask
        try:
            geometry_volume_cc = float(structure_entry.get("geometry_volume_cc", 0.0))
        except (TypeError, ValueError):
            geometry_volume_cc = 0.0
        if geometry_volume_cc > 0.0:
            structure_geometry_volumes_cc[normalized_name] = geometry_volume_cc

    rtstruct: Optional[RTStructData] = None
    if cache_version >= 2:
        rtstruct_geometry_payload = metadata.get("rtstruct_geometry")
        if rtstruct_geometry_payload not in (None, {}):
            if not isinstance(rtstruct_geometry_payload, dict):
                payload.close()
                return None
            rtstruct = _deserialize_rtstruct_geometry(rtstruct_geometry_payload, payload)
            if rtstruct is None:
                payload.close()
                return None

    payload.close()
    return DerivedArrayCacheData(
        rtstruct=rtstruct,
        sampled_dose_volume_ct=sampled_dose_volume_ct,
        ptv_union_volume_mask=ptv_union_volume_mask,
        structure_volume_masks=structure_volume_masks,
        structure_geometry_volumes_cc=structure_geometry_volumes_cc,
    )
