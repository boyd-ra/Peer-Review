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
    DoseVolume,
    DVHCurve,
    ImageViewBounds,
    PatientFileDiscovery,
    RTPlanPhase,
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


def get_review_bundle_path(current_patient_folder: Optional[str]) -> Optional[Path]:
    if current_patient_folder is None:
        return None
    return Path(current_patient_folder) / "peer_review_bundle.npz"


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


def write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
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
    load_ct_series_from_paths_func: Callable[..., object],
    load_combined_rtdose_func: Callable[..., object],
    load_rtstruct_func: Callable[..., object],
) -> Dict[str, str]:
    return {
        "sample_dose_to_ct_slice": callable_signature_hash(sample_dose_to_ct_slice_func),
        "build_structure_slice_mask": callable_signature_hash(build_structure_slice_mask_func),
        "get_target_structure_slice_masks": callable_signature_hash(get_target_structure_slice_masks_func),
        "get_ptv_union_slice_masks": callable_signature_hash(get_ptv_union_slice_masks_func),
        "load_ct_series_from_paths": callable_signature_hash(load_ct_series_from_paths_func),
        "load_combined_rtdose": callable_signature_hash(load_combined_rtdose_func),
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
    ct: Optional[CTVolume]
    dose: Optional[DoseVolume]
    rtstruct: Optional[RTStructData]
    patient_discovery: Optional[PatientFileDiscovery]
    image_view_bounds: Optional[ImageViewBounds]
    sampled_dose_volume_ct: Optional[np.ndarray]
    ptv_union_volume_mask: Optional[np.ndarray]
    structure_volume_masks: Dict[str, np.ndarray]
    structure_geometry_volumes_cc: Dict[str, float]


@dataclass(slots=True)
class ReviewCacheFileData:
    payload: Dict[str, object]
    cache_version: int
    trusted_source: bool = False


@dataclass(slots=True)
class ReviewBundleData:
    derived_array_cache_data: DerivedArrayCacheData
    review_cache_data: Optional[ReviewCacheFileData]
    screenshot_png_bytes: Optional[bytes]


@dataclass(slots=True)
class ReviewBundlePreviewData:
    patient_plan_lines: Optional[List[str]]
    screenshot_png_bytes: Optional[bytes]


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


def _serialize_patient_discovery(
    patient_discovery: Optional[PatientFileDiscovery],
    *,
    folder: Path,
) -> Dict[str, object]:
    if patient_discovery is None:
        return {}

    def _serialize_path(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        try:
            return os.path.relpath(path, folder)
        except ValueError:
            return str(path)

    return {
        "ct_paths": [serialized for path in patient_discovery.ct_paths if (serialized := _serialize_path(path))],
        "rtstruct_path": _serialize_path(patient_discovery.rtstruct_path),
        "rtdose_paths": [serialized for path in patient_discovery.rtdose_paths if (serialized := _serialize_path(path))],
        "rtplan_paths": [serialized for path in patient_discovery.rtplan_paths if (serialized := _serialize_path(path))],
        "plan_phases": [
            {
                "sop_instance_uid": phase.sop_instance_uid,
                "prescription_dose_gy": float(phase.prescription_dose_gy),
                "fractions_planned": int(phase.fractions_planned),
                "dose_path": _serialize_path(phase.dose_path),
                "target_structure_name": phase.target_structure_name,
                "plan_label": phase.plan_label,
                "plan_name": phase.plan_name,
            }
            for phase in patient_discovery.plan_phases
        ],
        "patient_plan_lines": list(patient_discovery.patient_plan_lines or []),
    }


def _deserialize_patient_discovery(
    payload: Mapping[str, object],
    *,
    folder: Path,
) -> Optional[PatientFileDiscovery]:
    def _deserialize_path(path_value: object) -> Optional[str]:
        if path_value in {None, ""}:
            return None
        path_text = str(path_value)
        return str((folder / path_text).resolve()) if not os.path.isabs(path_text) else path_text

    ct_payload = payload.get("ct_paths")
    rtdose_payload = payload.get("rtdose_paths")
    rtplan_payload = payload.get("rtplan_paths")
    if not isinstance(ct_payload, list) or not isinstance(rtdose_payload, list) or not isinstance(rtplan_payload, list):
        return None

    plan_phases_payload = payload.get("plan_phases", [])
    if not isinstance(plan_phases_payload, list):
        return None
    plan_phases: List[RTPlanPhase] = []
    for phase_payload in plan_phases_payload:
        if not isinstance(phase_payload, dict):
            return None
        plan_phases.append(
            RTPlanPhase(
                sop_instance_uid=str(phase_payload.get("sop_instance_uid", "")),
                prescription_dose_gy=float(phase_payload.get("prescription_dose_gy", 0.0) or 0.0),
                fractions_planned=int(phase_payload.get("fractions_planned", 0) or 0),
                dose_path=_deserialize_path(phase_payload.get("dose_path")) or "",
                target_structure_name=str(phase_payload.get("target_structure_name", "")),
                plan_label=str(phase_payload.get("plan_label", "")),
                plan_name=str(phase_payload.get("plan_name", "")),
            )
        )

    patient_plan_lines_payload = payload.get("patient_plan_lines")
    patient_plan_lines: Optional[Tuple[str, ...]]
    if isinstance(patient_plan_lines_payload, list):
        patient_plan_lines = tuple(str(line).strip() for line in patient_plan_lines_payload if str(line).strip()) or None
    else:
        patient_plan_lines = None

    return PatientFileDiscovery(
        ct_paths=[path for item in ct_payload if (path := _deserialize_path(item))],
        rtstruct_path=_deserialize_path(payload.get("rtstruct_path")),
        rtdose_paths=[path for item in rtdose_payload if (path := _deserialize_path(item))],
        rtplan_paths=[path for item in rtplan_payload if (path := _deserialize_path(item))],
        plan_phases=plan_phases,
        patient_plan_lines=patient_plan_lines,
    )


def _serialize_image_view_bounds(image_view_bounds: Optional[ImageViewBounds]) -> Dict[str, object]:
    if image_view_bounds is None:
        return {}
    return {
        "axial_by_slice": {
            str(int(slice_index)): [float(value) for value in bounds]
            for slice_index, bounds in image_view_bounds.axial_by_slice.items()
        },
        "sagittal": [float(value) for value in image_view_bounds.sagittal] if image_view_bounds.sagittal is not None else None,
        "coronal": [float(value) for value in image_view_bounds.coronal] if image_view_bounds.coronal is not None else None,
    }


def _deserialize_image_view_bounds(payload: Mapping[str, object]) -> Optional[ImageViewBounds]:
    axial_payload = payload.get("axial_by_slice")
    sagittal_payload = payload.get("sagittal")
    coronal_payload = payload.get("coronal")
    if not isinstance(axial_payload, dict):
        return None

    axial_by_slice: Dict[int, Tuple[float, float, float, float]] = {}
    for slice_index_text, bounds_payload in axial_payload.items():
        if not isinstance(bounds_payload, (list, tuple)) or len(bounds_payload) != 4:
            continue
        try:
            axial_by_slice[int(slice_index_text)] = tuple(float(value) for value in bounds_payload)
        except (TypeError, ValueError):
            continue

    def _deserialize_bounds(bounds_payload: object) -> Optional[Tuple[float, float, float, float]]:
        if not isinstance(bounds_payload, (list, tuple)) or len(bounds_payload) != 4:
            return None
        try:
            return tuple(float(value) for value in bounds_payload)
        except (TypeError, ValueError):
            return None

    sagittal = _deserialize_bounds(sagittal_payload)
    coronal = _deserialize_bounds(coronal_payload)
    if not axial_by_slice and sagittal is None and coronal is None:
        return None
    return ImageViewBounds(axial_by_slice=axial_by_slice, sagittal=sagittal, coronal=coronal)


def load_cached_patient_discovery(path: Path, *, folder: str) -> Optional[PatientFileDiscovery]:
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
    if cache_version not in {5}:
        payload.close()
        return None

    patient_discovery_payload = metadata.get("patient_discovery")
    if not isinstance(patient_discovery_payload, dict):
        payload.close()
        return None

    patient_discovery = _deserialize_patient_discovery(patient_discovery_payload, folder=Path(folder))
    if patient_discovery is None:
        payload.close()
        return None

    payload.close()

    if not file_fingerprint_list_matches(metadata.get("ct_fingerprints"), list(patient_discovery.ct_paths)):
        return None
    if not file_fingerprint_matches(metadata.get("rtstruct_fingerprint"), patient_discovery.rtstruct_path):
        return None
    if not file_fingerprint_list_matches(metadata.get("rtdose_fingerprints"), list(patient_discovery.rtdose_paths)):
        return None
    if not file_fingerprint_list_matches(metadata.get("rtplan_fingerprints"), list(patient_discovery.rtplan_paths)):
        return None
    return patient_discovery


def _serialize_ct_geometry(
    ct: Optional[CTVolume],
    arrays: Dict[str, np.ndarray],
) -> Dict[str, object]:
    if ct is None:
        return {}

    arrays["ct_volume_hu"] = np.asarray(ct.volume_hu, dtype=np.float32)
    arrays["ct_slice_origins_xyz_mm"] = np.asarray(ct.slice_origins_xyz_mm, dtype=np.float32)
    arrays["ct_z_positions_mm"] = np.asarray(ct.z_positions_mm, dtype=np.float32)
    arrays["ct_spacing_xyz_mm"] = np.asarray(ct.spacing_xyz_mm, dtype=np.float32)
    arrays["ct_image_orientation_patient"] = np.asarray(ct.image_orientation_patient, dtype=np.float32)
    return {
        "study_uid": ct.study_uid,
        "frame_of_reference_uid": ct.frame_of_reference_uid,
        "rows": int(ct.rows),
        "cols": int(ct.cols),
    }


def _deserialize_ct_geometry(
    payload: Mapping[str, object],
    arrays: Any,
) -> Optional[CTVolume]:
    try:
        volume_hu = np.asarray(arrays["ct_volume_hu"], dtype=np.float32)
        slice_origins_xyz_mm = np.asarray(arrays["ct_slice_origins_xyz_mm"], dtype=np.float32)
        z_positions_mm = np.asarray(arrays["ct_z_positions_mm"], dtype=np.float32)
        spacing_xyz_mm = np.asarray(arrays["ct_spacing_xyz_mm"], dtype=np.float32)
        image_orientation_patient = np.asarray(arrays["ct_image_orientation_patient"], dtype=np.float32)
    except KeyError:
        return None

    if volume_hu.ndim != 3:
        return None
    if slice_origins_xyz_mm.shape != (volume_hu.shape[0], 3):
        return None
    if z_positions_mm.shape != (volume_hu.shape[0],):
        return None
    if spacing_xyz_mm.shape != (3,):
        return None
    if image_orientation_patient.shape != (6,):
        return None

    try:
        rows = int(payload.get("rows", volume_hu.shape[1]))
        cols = int(payload.get("cols", volume_hu.shape[2]))
    except (TypeError, ValueError):
        return None
    if rows != volume_hu.shape[1] or cols != volume_hu.shape[2]:
        return None

    return CTVolume(
        volume_hu=volume_hu.copy(),
        slice_origins_xyz_mm=slice_origins_xyz_mm.copy(),
        z_positions_mm=z_positions_mm.copy(),
        spacing_xyz_mm=spacing_xyz_mm.copy(),
        image_orientation_patient=image_orientation_patient.copy(),
        study_uid=str(payload.get("study_uid", "")),
        frame_of_reference_uid=str(payload.get("frame_of_reference_uid", "")),
        rows=rows,
        cols=cols,
    )


def _serialize_dose_geometry(
    dose: Optional[DoseVolume],
    arrays: Dict[str, np.ndarray],
) -> Dict[str, object]:
    if dose is None:
        return {}

    arrays["dose_dose_gy"] = np.asarray(dose.dose_gy, dtype=np.float32)
    arrays["dose_slice_origins_xyz_mm"] = np.asarray(dose.slice_origins_xyz_mm, dtype=np.float32)
    arrays["dose_z_positions_mm"] = np.asarray(dose.z_positions_mm, dtype=np.float32)
    arrays["dose_origin_xyz_mm"] = np.asarray(dose.origin_xyz_mm, dtype=np.float32)
    arrays["dose_spacing_xyz_mm"] = np.asarray(dose.spacing_xyz_mm, dtype=np.float32)
    arrays["dose_image_orientation_patient"] = np.asarray(dose.image_orientation_patient, dtype=np.float32)
    return {
        "frame_of_reference_uid": dose.frame_of_reference_uid,
        "dose_units": dose.dose_units,
    }


def _deserialize_dose_geometry(
    payload: Mapping[str, object],
    arrays: Any,
) -> Optional[DoseVolume]:
    try:
        dose_gy = np.asarray(arrays["dose_dose_gy"], dtype=np.float32)
        slice_origins_xyz_mm = np.asarray(arrays["dose_slice_origins_xyz_mm"], dtype=np.float32)
        z_positions_mm = np.asarray(arrays["dose_z_positions_mm"], dtype=np.float32)
        origin_xyz_mm = np.asarray(arrays["dose_origin_xyz_mm"], dtype=np.float32)
        spacing_xyz_mm = np.asarray(arrays["dose_spacing_xyz_mm"], dtype=np.float32)
        image_orientation_patient = np.asarray(arrays["dose_image_orientation_patient"], dtype=np.float32)
    except KeyError:
        return None

    if dose_gy.ndim != 3:
        return None
    if slice_origins_xyz_mm.shape != (dose_gy.shape[0], 3):
        return None
    if z_positions_mm.shape != (dose_gy.shape[0],):
        return None
    if origin_xyz_mm.shape != (3,):
        return None
    if spacing_xyz_mm.shape != (3,):
        return None
    if image_orientation_patient.shape != (6,):
        return None

    return DoseVolume(
        dose_gy=dose_gy.copy(),
        slice_origins_xyz_mm=slice_origins_xyz_mm.copy(),
        z_positions_mm=z_positions_mm.copy(),
        origin_xyz_mm=origin_xyz_mm.copy(),
        spacing_xyz_mm=spacing_xyz_mm.copy(),
        image_orientation_patient=image_orientation_patient.copy(),
        frame_of_reference_uid=str(payload.get("frame_of_reference_uid", "")),
        dose_units=str(payload.get("dose_units", "")),
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
    constraint_script_xml_path: Optional[str],
    script_constraints_label: Optional[str],
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
    trusted_source = loaded_cache.trusted_source

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

    if not trusted_source and saved_constraints_sheet is not None and saved_constraints_sheet not in available_constraint_sheet_names:
        return None

    current_constraints_source_path = (
        constraint_script_xml_path
        if script_constraints_label and saved_constraints_sheet == script_constraints_label
        else structure_filter_csv_path
    )

    saved_csv_fingerprint = payload.get("constraints_fingerprint", payload.get("csv_fingerprint"))
    if (
        not trusted_source
        and saved_csv_fingerprint is not None
        and not file_fingerprint_matches(saved_csv_fingerprint, current_constraints_source_path)
    ):
        return None

    saved_csv_name = payload.get("constraints_file", payload.get("csv_file"))
    current_csv_name = Path(current_constraints_source_path).name if current_constraints_source_path else None
    if not trusted_source and saved_csv_fingerprint is None and saved_csv_name not in {None, current_csv_name}:
        return None

    saved_rtstruct_fingerprint = payload.get("rtstruct_fingerprint")
    if (
        not trusted_source
        and saved_rtstruct_fingerprint is not None
        and not file_fingerprint_matches(saved_rtstruct_fingerprint, rtstruct_path)
    ):
        return None

    saved_rtstruct_name = payload.get("rtstruct_file")
    current_rtstruct_name = Path(rtstruct_path).name if rtstruct_path else None
    if not trusted_source and saved_rtstruct_fingerprint is None and saved_rtstruct_name not in {None, current_rtstruct_name}:
        return None

    saved_rtdose_fingerprints = payload.get("rtdose_fingerprints")
    if (
        not trusted_source
        and saved_rtdose_fingerprints is not None
        and not file_fingerprint_list_matches(saved_rtdose_fingerprints, list(rtdose_paths))
    ):
        return None

    saved_rtplan_fingerprints = payload.get("rtplan_fingerprints")
    if (
        not trusted_source
        and saved_rtplan_fingerprints is not None
        and not file_fingerprint_list_matches(saved_rtplan_fingerprints, list(rtplan_paths))
    ):
        return None

    saved_dvh_mode = payload.get("dvh_mode")
    if not trusted_source and saved_dvh_mode not in {None, dvh_mode}:
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
    if not trusted_source and saved_dvh_signature is not None and saved_dvh_signature != dvh_method_signature:
        return None

    saved_target_signature = payload.get("target_method_signature")
    if not trusted_source and saved_target_signature is not None and saved_target_signature != target_method_signature:
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


def build_derived_array_archive(
    *,
    folder: Path,
    ct: CTVolume,
    ct_paths: Sequence[str],
    patient_discovery: Optional[PatientFileDiscovery],
    image_view_bounds: Optional[ImageViewBounds],
    dose: Optional[DoseVolume],
    rtstruct: Optional[RTStructData],
    rtstruct_path: Optional[str],
    rtdose_paths: Sequence[str],
    array_cache_signature: Dict[str, str],
    sampled_dose_volume_ct: Optional[np.ndarray],
    ptv_union_volume_mask: Optional[np.ndarray],
    structure_order: Sequence[str],
    structure_volume_masks: Dict[str, np.ndarray],
    structure_geometry_volumes_cc: Dict[str, float],
    metadata_overrides: Optional[Mapping[str, object]] = None,
    extra_arrays: Optional[Mapping[str, np.ndarray]] = None,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    metadata: Dict[str, object] = {
        "version": 5,
        "saved_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "ct_geometry_signature": get_ct_geometry_signature(ct),
        "ct_fingerprints": build_file_fingerprints(list(ct_paths)),
        "rtstruct_fingerprint": build_file_fingerprint(rtstruct_path),
        "rtdose_fingerprints": build_file_fingerprints(list(rtdose_paths)),
        "rtplan_fingerprints": build_file_fingerprints(
            list(patient_discovery.rtplan_paths if patient_discovery is not None else [])
        ),
        "array_cache_signature": array_cache_signature,
        "structures": [],
        "ct_geometry": {},
        "dose_geometry": {},
        "rtstruct_geometry": {},
        "patient_discovery": _serialize_patient_discovery(patient_discovery, folder=folder),
        "image_view_bounds": _serialize_image_view_bounds(image_view_bounds),
    }

    arrays: Dict[str, np.ndarray] = {}
    metadata["ct_geometry"] = _serialize_ct_geometry(ct, arrays)
    metadata["dose_geometry"] = _serialize_dose_geometry(dose, arrays)
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

    if metadata_overrides:
        metadata.update(dict(metadata_overrides))
    if extra_arrays:
        arrays.update(dict(extra_arrays))

    metadata["patient_discovery"] = _serialize_patient_discovery(patient_discovery, folder=folder)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata), dtype=np.str_)
    return metadata, arrays


def save_derived_array_cache(
    path: Path,
    *,
    ct: CTVolume,
    ct_paths: Sequence[str],
    patient_discovery: Optional[PatientFileDiscovery],
    image_view_bounds: Optional[ImageViewBounds],
    dose: Optional[DoseVolume],
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
    _metadata, arrays = build_derived_array_archive(
        folder=path.parent,
        ct=ct,
        ct_paths=ct_paths,
        patient_discovery=patient_discovery,
        image_view_bounds=image_view_bounds,
        dose=dose,
        rtstruct=rtstruct,
        rtstruct_path=rtstruct_path,
        rtdose_paths=rtdose_paths,
        array_cache_signature=array_cache_signature,
        sampled_dose_volume_ct=sampled_dose_volume_ct,
        ptv_union_volume_mask=ptv_union_volume_mask,
        structure_order=structure_order,
        structure_volume_masks=structure_volume_masks,
        structure_geometry_volumes_cc=structure_geometry_volumes_cc,
    )
    write_npz_atomic(path, arrays)


def save_review_bundle(
    path: Path,
    *,
    review_payload: Mapping[str, object],
    screenshot_png_bytes: Optional[bytes],
    ct: CTVolume,
    ct_paths: Sequence[str],
    patient_discovery: Optional[PatientFileDiscovery],
    image_view_bounds: Optional[ImageViewBounds],
    dose: Optional[DoseVolume],
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
    preview_payload = {
        "patient_plan_lines": list(review_payload.get("patient_plan_lines", []) or []),
    }
    extra_arrays: Dict[str, np.ndarray] = {
        "review_cache_json": np.asarray(json.dumps(dict(review_payload)), dtype=np.str_),
        "preview_json": np.asarray(json.dumps(preview_payload), dtype=np.str_),
    }
    if screenshot_png_bytes:
        extra_arrays["screenshot_png_bytes"] = np.frombuffer(screenshot_png_bytes, dtype=np.uint8)

    _metadata, arrays = build_derived_array_archive(
        folder=path.parent,
        ct=ct,
        ct_paths=ct_paths,
        patient_discovery=patient_discovery,
        image_view_bounds=image_view_bounds,
        dose=dose,
        rtstruct=rtstruct,
        rtstruct_path=rtstruct_path,
        rtdose_paths=rtdose_paths,
        array_cache_signature=array_cache_signature,
        sampled_dose_volume_ct=sampled_dose_volume_ct,
        ptv_union_volume_mask=ptv_union_volume_mask,
        structure_order=structure_order,
        structure_volume_masks=structure_volume_masks,
        structure_geometry_volumes_cc=structure_geometry_volumes_cc,
        metadata_overrides={
            "cache_kind": "peer_review_bundle",
            "bundle_version": 1,
            "review_cache_version": int(review_payload.get("version", 0) or 0),
        },
        extra_arrays=extra_arrays,
    )
    write_npz_atomic(path, arrays)


def load_derived_array_cache(
    path: Path,
    *,
    ct: Optional[CTVolume],
    ct_paths: Sequence[str],
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
    if cache_version not in {1, 2, 3, 4, 5}:
        payload.close()
        return None
    if metadata.get("array_cache_signature") != array_cache_signature:
        payload.close()
        return None
    if cache_version >= 3:
        if not file_fingerprint_list_matches(metadata.get("ct_fingerprints"), list(ct_paths)):
            payload.close()
            return None
    if not file_fingerprint_matches(metadata.get("rtstruct_fingerprint"), rtstruct_path):
        payload.close()
        return None
    if not file_fingerprint_list_matches(metadata.get("rtdose_fingerprints"), list(rtdose_paths)):
        payload.close()
        return None

    loaded_ct = ct
    if cache_version >= 3:
        ct_geometry_payload = metadata.get("ct_geometry")
        if ct_geometry_payload not in (None, {}):
            if not isinstance(ct_geometry_payload, dict):
                payload.close()
                return None
            loaded_ct = _deserialize_ct_geometry(ct_geometry_payload, payload)
            if loaded_ct is None:
                payload.close()
                return None

    if loaded_ct is None:
        payload.close()
        return None
    if metadata.get("ct_geometry_signature") != get_ct_geometry_signature(loaded_ct):
        payload.close()
        return None
    if ct is not None and get_ct_geometry_signature(ct) != get_ct_geometry_signature(loaded_ct):
        payload.close()
        return None

    dose: Optional[DoseVolume] = None
    if cache_version >= 4:
        dose_geometry_payload = metadata.get("dose_geometry")
        if dose_geometry_payload not in (None, {}):
            if not isinstance(dose_geometry_payload, dict):
                payload.close()
                return None
            dose = _deserialize_dose_geometry(dose_geometry_payload, payload)
            if dose is None:
                payload.close()
                return None

    sampled_dose_volume_ct: Optional[np.ndarray] = None
    if "sampled_dose_volume_ct" in payload.files:
        sampled_dose_volume_ct = np.asarray(payload["sampled_dose_volume_ct"], dtype=np.float32)
        if sampled_dose_volume_ct.shape != loaded_ct.volume_hu.shape:
            payload.close()
            return None

    ptv_union_volume_mask: Optional[np.ndarray] = None
    if "ptv_union_volume_mask" in payload.files:
        ptv_union_volume_mask = np.asarray(payload["ptv_union_volume_mask"], dtype=bool)
        if ptv_union_volume_mask.shape != loaded_ct.volume_hu.shape:
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
        if cached_mask.shape != loaded_ct.volume_hu.shape:
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

    patient_discovery: Optional[PatientFileDiscovery] = None
    if cache_version >= 5:
        patient_discovery_payload = metadata.get("patient_discovery")
        if patient_discovery_payload not in (None, {}):
            if not isinstance(patient_discovery_payload, dict):
                payload.close()
                return None
            patient_discovery = _deserialize_patient_discovery(patient_discovery_payload, folder=path.parent)
            if patient_discovery is None:
                payload.close()
                return None

    image_view_bounds: Optional[ImageViewBounds] = None
    image_view_bounds_payload = metadata.get("image_view_bounds")
    if image_view_bounds_payload not in (None, {}):
        if not isinstance(image_view_bounds_payload, dict):
            payload.close()
            return None
        image_view_bounds = _deserialize_image_view_bounds(image_view_bounds_payload)

    payload.close()
    return DerivedArrayCacheData(
        ct=loaded_ct,
        dose=dose,
        rtstruct=rtstruct,
        patient_discovery=patient_discovery,
        image_view_bounds=image_view_bounds,
        sampled_dose_volume_ct=sampled_dose_volume_ct,
        ptv_union_volume_mask=ptv_union_volume_mask,
        structure_volume_masks=structure_volume_masks,
        structure_geometry_volumes_cc=structure_geometry_volumes_cc,
    )


def load_review_bundle(path: Path) -> Optional[ReviewBundleData]:
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
    if metadata.get("cache_kind") != "peer_review_bundle":
        payload.close()
        return None
    if metadata.get("bundle_version") not in {1}:
        payload.close()
        return None

    ct_geometry_payload = metadata.get("ct_geometry")
    if not isinstance(ct_geometry_payload, dict):
        payload.close()
        return None
    loaded_ct = _deserialize_ct_geometry(ct_geometry_payload, payload)
    if loaded_ct is None:
        payload.close()
        return None

    dose: Optional[DoseVolume] = None
    dose_geometry_payload = metadata.get("dose_geometry")
    if dose_geometry_payload not in (None, {}):
        if not isinstance(dose_geometry_payload, dict):
            payload.close()
            return None
        dose = _deserialize_dose_geometry(dose_geometry_payload, payload)
        if dose is None:
            payload.close()
            return None

    sampled_dose_volume_ct: Optional[np.ndarray] = None
    if "sampled_dose_volume_ct" in payload.files:
        sampled_dose_volume_ct = np.asarray(payload["sampled_dose_volume_ct"], dtype=np.float32)
        if sampled_dose_volume_ct.shape != loaded_ct.volume_hu.shape:
            payload.close()
            return None

    ptv_union_volume_mask: Optional[np.ndarray] = None
    if "ptv_union_volume_mask" in payload.files:
        ptv_union_volume_mask = np.asarray(payload["ptv_union_volume_mask"], dtype=bool)
        if ptv_union_volume_mask.shape != loaded_ct.volume_hu.shape:
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
        if cached_mask.shape != loaded_ct.volume_hu.shape:
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
    rtstruct_geometry_payload = metadata.get("rtstruct_geometry")
    if rtstruct_geometry_payload not in (None, {}):
        if not isinstance(rtstruct_geometry_payload, dict):
            payload.close()
            return None
        rtstruct = _deserialize_rtstruct_geometry(rtstruct_geometry_payload, payload)
        if rtstruct is None:
            payload.close()
            return None

    patient_discovery: Optional[PatientFileDiscovery] = None
    patient_discovery_payload = metadata.get("patient_discovery")
    if patient_discovery_payload not in (None, {}):
        if not isinstance(patient_discovery_payload, dict):
            payload.close()
            return None
        patient_discovery = _deserialize_patient_discovery(patient_discovery_payload, folder=path.parent)
        if patient_discovery is None:
            payload.close()
            return None

    image_view_bounds: Optional[ImageViewBounds] = None
    image_view_bounds_payload = metadata.get("image_view_bounds")
    if image_view_bounds_payload not in (None, {}):
        if not isinstance(image_view_bounds_payload, dict):
            payload.close()
            return None
        image_view_bounds = _deserialize_image_view_bounds(image_view_bounds_payload)

    review_cache_data: Optional[ReviewCacheFileData] = None
    if "review_cache_json" in payload.files:
        try:
            review_cache_json = payload["review_cache_json"].item()
            review_payload = json.loads(str(review_cache_json))
        except (ValueError, TypeError, json.JSONDecodeError):
            payload.close()
            return None
        if not isinstance(review_payload, dict):
            payload.close()
            return None
        try:
            cache_version = int(review_payload.get("version", metadata.get("review_cache_version", 0)))
        except (TypeError, ValueError):
            cache_version = 0
        review_cache_data = ReviewCacheFileData(
            payload=review_payload,
            cache_version=cache_version,
            trusted_source=True,
        )

    screenshot_png_bytes: Optional[bytes] = None
    if "screenshot_png_bytes" in payload.files:
        screenshot_png_bytes = bytes(np.asarray(payload["screenshot_png_bytes"], dtype=np.uint8).tobytes())
    payload.close()

    return ReviewBundleData(
        derived_array_cache_data=DerivedArrayCacheData(
            ct=loaded_ct,
            dose=dose,
            rtstruct=rtstruct,
            patient_discovery=patient_discovery,
            image_view_bounds=image_view_bounds,
            sampled_dose_volume_ct=sampled_dose_volume_ct,
            ptv_union_volume_mask=ptv_union_volume_mask,
            structure_volume_masks=structure_volume_masks,
            structure_geometry_volumes_cc=structure_geometry_volumes_cc,
        ),
        review_cache_data=review_cache_data,
        screenshot_png_bytes=screenshot_png_bytes,
    )


def load_review_bundle_preview(path: Path) -> Optional[ReviewBundlePreviewData]:
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

    if (
        not isinstance(metadata, dict)
        or metadata.get("cache_kind") != "peer_review_bundle"
        or metadata.get("bundle_version") not in {1}
    ):
        payload.close()
        return None

    patient_plan_lines: Optional[List[str]] = None
    if "preview_json" in payload.files:
        try:
            preview_json = payload["preview_json"].item()
            preview_payload = json.loads(str(preview_json))
        except (ValueError, TypeError, json.JSONDecodeError):
            payload.close()
            return None
        if not isinstance(preview_payload, dict):
            payload.close()
            return None
        patient_plan_lines_payload = preview_payload.get("patient_plan_lines")
        if isinstance(patient_plan_lines_payload, list):
            patient_plan_lines = [
                str(line).strip()
                for line in patient_plan_lines_payload
                if str(line).strip()
            ]
    elif "review_cache_json" in payload.files:
        try:
            review_cache_json = payload["review_cache_json"].item()
            review_payload = json.loads(str(review_cache_json))
        except (ValueError, TypeError, json.JSONDecodeError):
            payload.close()
            return None
        if isinstance(review_payload, dict):
            patient_plan_lines_payload = review_payload.get("patient_plan_lines")
            if isinstance(patient_plan_lines_payload, list):
                patient_plan_lines = [
                    str(line).strip()
                    for line in patient_plan_lines_payload
                    if str(line).strip()
                ]

    screenshot_png_bytes: Optional[bytes] = None
    if "screenshot_png_bytes" in payload.files:
        screenshot_png_bytes = bytes(np.asarray(payload["screenshot_png_bytes"], dtype=np.uint8).tobytes())
    payload.close()
    return ReviewBundlePreviewData(
        patient_plan_lines=patient_plan_lines,
        screenshot_png_bytes=screenshot_png_bytes,
    )
