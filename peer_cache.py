from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from peer_helpers import normalize_structure_name
from peer_models import CTVolume, RTStructData, StructureSliceContours
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
