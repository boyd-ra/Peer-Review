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
) -> Dict[str, str]:
    return {
        "sample_dose_to_ct_slice": callable_signature_hash(sample_dose_to_ct_slice_func),
        "build_structure_slice_mask": callable_signature_hash(build_structure_slice_mask_func),
        "get_target_structure_slice_masks": callable_signature_hash(get_target_structure_slice_masks_func),
        "get_ptv_union_slice_masks": callable_signature_hash(get_ptv_union_slice_masks_func),
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


@dataclass(slots=True)
class DerivedArrayCacheData:
    sampled_dose_volume_ct: Optional[np.ndarray]
    ptv_union_volume_mask: Optional[np.ndarray]
    structure_volume_masks: Dict[str, np.ndarray]
    structure_geometry_volumes_cc: Dict[str, float]


@dataclass(slots=True)
class ReviewCacheFileData:
    payload: Dict[str, object]
    cache_version: int


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
    hidden_structure_names: Sequence[str],
    additional_target_subvolume_names: Sequence[str],
    constraint_notes: Mapping[str, str],
    target_notes: Mapping[str, str],
) -> Dict[str, object]:
    return {
        "version": 15,
        "saved_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
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


def save_derived_array_cache(
    path: Path,
    *,
    ct: CTVolume,
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
        "version": 1,
        "saved_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "ct_geometry_signature": get_ct_geometry_signature(ct),
        "rtstruct_fingerprint": build_file_fingerprint(rtstruct_path),
        "rtdose_fingerprints": build_file_fingerprints(list(rtdose_paths)),
        "array_cache_signature": array_cache_signature,
        "structures": [],
    }

    arrays: Dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata), dtype=np.str_),
    }
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
    if metadata.get("version") != 1:
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

    payload.close()
    return DerivedArrayCacheData(
        sampled_dose_volume_ct=sampled_dose_volume_ct,
        ptv_union_volume_mask=ptv_union_volume_mask,
        structure_volume_masks=structure_volume_masks,
        structure_geometry_volumes_cc=structure_geometry_volumes_cc,
    )
