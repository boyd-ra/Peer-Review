from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import multiprocessing
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore

from peer_activation_worker import prepare_activation_review_cache_state
from peer_cache import (
    DerivedArrayCacheData,
    PreparedReviewCacheState,
    ReviewCacheFileData,
    get_derived_array_cache_path,
    get_dvh_cache_path,
    load_derived_array_cache,
    load_review_cache_file,
)
from peer_helpers import compute_image_view_bounds, normalize_structure_name, sample_dose_to_ct_slice
from peer_io import load_combined_rtdose, load_ct_series_and_discover_patient_files, load_rtstruct
from peer_models import CTVolume, DoseVolume, ImageViewBounds, PatientFileDiscovery, RTPlanPhase, RTStructData
from peer_rendering import (
    AxialRenderState,
    OrthogonalRenderState,
    build_axial_render_state,
    build_orthogonal_render_state,
)


@dataclass(slots=True)
class ReviewCacheAvailability:
    dvh_can_start: bool
    cache_path: Optional[Path]
    cache_found: bool
    derived_sidecar_only: bool


@dataclass(slots=True)
class PatientPreloadPayload:
    folder: str
    ct: CTVolume
    image_view_bounds: ImageViewBounds
    patient_plan_lines: Optional[List[str]]
    plan_phases: List[RTPlanPhase]
    rtplan_paths: List[str]
    rtstruct_path: Optional[str]
    rtstruct: Optional[RTStructData]
    rtdose_paths: List[str]
    dose: Optional[DoseVolume]
    sampled_dose_volume_ct: Optional[np.ndarray]
    derived_array_cache_loaded: bool
    derived_array_cache_data: Optional[DerivedArrayCacheData]
    review_cache_data: Optional[ReviewCacheFileData]
    precomputed_view_state: Optional["PrecomputedPatientViewState"]
    timing_entries: List[Tuple[str, Optional[float]]]


@dataclass(slots=True)
class PrecomputedPatientViewState:
    slice_index: int
    row_idx: int
    col_idx: int
    window_level: float
    window_width: float
    dose_alpha: float
    dose_min_gy: float
    dose_max_gy: float
    axial_render_state: AxialRenderState
    orthogonal_render_state: OrthogonalRenderState


@dataclass(slots=True)
class PatientActivationPreparationPayload:
    prepared_review_cache_state: Optional[PreparedReviewCacheState]
    cache_loaded: bool
    cache_load_duration: Optional[float]
    used_preloaded_review_cache: bool


class PatientPreloadSignals(QtCore.QObject):
    finished = QtCore.Signal(int, str, object, float)
    failed = QtCore.Signal(int, str, str, float)


class PatientActivationPreparationSignals(QtCore.QObject):
    finished = QtCore.Signal(int, str, object, float)
    failed = QtCore.Signal(int, str, str, float)


class PatientPreloadTask(QtCore.QRunnable):
    def __init__(
        self,
        request_id: int,
        folder: str,
        array_cache_signature: Dict[str, str],
    ):
        super().__init__()
        self.setAutoDelete(False)
        self.request_id = request_id
        self.folder = folder
        self.array_cache_signature = dict(array_cache_signature)
        self.signals = PatientPreloadSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        start = perf_counter()
        try:
            payload = prepare_patient_preload_payload(
                self.folder,
                array_cache_signature=self.array_cache_signature,
                include_precomputed_view_state=True,
            )
        except Exception as exc:
            if self._cancelled:
                return
            try:
                self.signals.failed.emit(self.request_id, self.folder, str(exc), perf_counter() - start)
            except RuntimeError:
                pass
            return
        if self._cancelled:
            return
        try:
            self.signals.finished.emit(self.request_id, self.folder, payload, perf_counter() - start)
        except RuntimeError:
            pass


class PatientActivationPreparationTask(QtCore.QRunnable):
    def __init__(
        self,
        request_id: int,
        folder: str,
        *,
        review_cache_data: Optional[ReviewCacheFileData],
        expected_structure_names: List[str],
        available_constraint_sheet_names: List[str],
        no_constraints_sheet_label: str,
        constraints_sheet_name: Optional[str],
        structure_filter_csv_path: Optional[str],
        rtstruct_path: Optional[str],
        rtdose_paths: List[str],
        rtplan_paths: List[str],
        dvh_mode: str,
        dvh_method_signature: str,
        target_method_signature: Dict[str, object],
        has_ct: bool,
        has_dose: bool,
        is_base_listable_structure_name: Callable[[str], bool],
    ):
        super().__init__()
        self.setAutoDelete(False)
        self.request_id = request_id
        self.folder = folder
        self.review_cache_data = review_cache_data
        self.expected_structure_names = list(expected_structure_names)
        self.available_constraint_sheet_names = list(available_constraint_sheet_names)
        self.no_constraints_sheet_label = no_constraints_sheet_label
        self.constraints_sheet_name = constraints_sheet_name
        self.structure_filter_csv_path = structure_filter_csv_path
        self.rtstruct_path = rtstruct_path
        self.rtdose_paths = list(rtdose_paths)
        self.rtplan_paths = list(rtplan_paths)
        self.dvh_mode = dvh_mode
        self.dvh_method_signature = dvh_method_signature
        self.target_method_signature = dict(target_method_signature)
        self.has_ct = has_ct
        self.has_dose = has_dose
        self.is_base_listable_structure_name = is_base_listable_structure_name
        self.signals = PatientActivationPreparationSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        start = perf_counter()
        try:
            cache_loaded = False
            cache_load_duration: Optional[float] = None
            prepared_state: Optional[PreparedReviewCacheState] = None
            used_preloaded_review_cache = False
            if self.review_cache_data is not None:
                stage_start = perf_counter()
                prepared_state = prepare_review_cache_state(
                    self.review_cache_data,
                    expected_structure_names=self.expected_structure_names,
                    available_constraint_sheet_names=self.available_constraint_sheet_names,
                    no_constraints_sheet_label=self.no_constraints_sheet_label,
                    constraints_sheet_name=self.constraints_sheet_name,
                    structure_filter_csv_path=self.structure_filter_csv_path,
                    rtstruct_path=self.rtstruct_path,
                    rtdose_paths=self.rtdose_paths,
                    rtplan_paths=self.rtplan_paths,
                    dvh_mode=self.dvh_mode,
                    dvh_method_signature=self.dvh_method_signature,
                    target_method_signature=self.target_method_signature,
                    has_ct=self.has_ct,
                    has_dose=self.has_dose,
                    is_base_listable_structure_name=self.is_base_listable_structure_name,
                )
                if prepared_state is not None:
                    cache_loaded = True
                    used_preloaded_review_cache = True
                    cache_load_duration = perf_counter() - stage_start
            payload = PatientActivationPreparationPayload(
                prepared_review_cache_state=prepared_state,
                cache_loaded=cache_loaded,
                cache_load_duration=cache_load_duration,
                used_preloaded_review_cache=used_preloaded_review_cache,
            )
        except Exception as exc:
            if self._cancelled:
                return
            try:
                self.signals.failed.emit(self.request_id, self.folder, str(exc), perf_counter() - start)
            except RuntimeError:
                pass
            return
        if self._cancelled:
            return
        try:
            self.signals.finished.emit(self.request_id, self.folder, payload, perf_counter() - start)
        except RuntimeError:
            pass


class PatientPreloadManager(QtCore.QObject):
    finished = QtCore.Signal(int, str, object, float)
    failed = QtCore.Signal(int, str, str, float)

    def __init__(self, thread_pool: Optional[QtCore.QThreadPool] = None):
        super().__init__()
        self.thread_pool = thread_pool or QtCore.QThreadPool.globalInstance()
        self.request_id = 0
        self.active_jobs: Dict[int, PatientPreloadTask] = {}

    def is_current(self, request_id: int) -> bool:
        return request_id == self.request_id

    def invalidate(self) -> None:
        self.request_id += 1

    def cancel_all(self) -> None:
        self.request_id += 1
        for task in self.active_jobs.values():
            task.cancel()

    def start(self, folder: str, *, array_cache_signature: Dict[str, str]) -> int:
        self.request_id += 1
        request_id = self.request_id
        task = PatientPreloadTask(request_id, folder, array_cache_signature)
        task.signals.finished.connect(self._on_task_finished)
        task.signals.failed.connect(self._on_task_failed)
        self.active_jobs[request_id] = task
        self.thread_pool.start(task)
        return request_id

    def _on_task_finished(
        self,
        request_id: int,
        folder: str,
        payload: object,
        duration_s: float,
    ) -> None:
        self.active_jobs.pop(request_id, None)
        self.finished.emit(request_id, folder, payload, duration_s)

    def _on_task_failed(
        self,
        request_id: int,
        folder: str,
        error_message: str,
        duration_s: float,
    ) -> None:
        self.active_jobs.pop(request_id, None)
        self.failed.emit(request_id, folder, error_message, duration_s)


class PatientActivationPreparationManager(QtCore.QObject):
    finished = QtCore.Signal(int, str, object, float)
    failed = QtCore.Signal(int, str, str, float)

    def __init__(self):
        super().__init__()
        self.request_id = 0
        self.active_jobs: Dict[int, Tuple[str, Future[Optional[PreparedReviewCacheState]], float]] = {}
        self.executor: Optional[ProcessPoolExecutor] = None
        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.setInterval(50)
        self.poll_timer.timeout.connect(self._poll_jobs)

    def cancel_all(self) -> None:
        self.request_id += 1
        for _folder, future, _started_at in self.active_jobs.values():
            future.cancel()
        self.active_jobs = {}
        self.poll_timer.stop()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None

    def start(self, folder: str, **kwargs: object) -> int:
        self.request_id += 1
        request_id = self.request_id
        if self.executor is None:
            self.executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )
        future = self.executor.submit(
            prepare_activation_review_cache_state,
            kwargs.get("review_cache_data"),
            expected_structure_names=kwargs.get("expected_structure_names", []),
            available_constraint_sheet_names=kwargs.get("available_constraint_sheet_names", []),
            no_constraints_sheet_label=kwargs.get("no_constraints_sheet_label", "------"),
            constraints_sheet_name=kwargs.get("constraints_sheet_name"),
            structure_filter_csv_path=kwargs.get("structure_filter_csv_path"),
            rtstruct_path=kwargs.get("rtstruct_path"),
            rtdose_paths=kwargs.get("rtdose_paths", []),
            rtplan_paths=kwargs.get("rtplan_paths", []),
            dvh_mode=kwargs.get("dvh_mode", ""),
            dvh_method_signature=kwargs.get("dvh_method_signature", ""),
            target_method_signature=kwargs.get("target_method_signature", {}),
            has_ct=bool(kwargs.get("has_ct", False)),
            has_dose=bool(kwargs.get("has_dose", False)),
        )
        self.active_jobs[request_id] = (folder, future, perf_counter())
        if not self.poll_timer.isActive():
            self.poll_timer.start()
        return request_id

    def _poll_jobs(self) -> None:
        for request_id, (folder, future, started_at) in list(self.active_jobs.items()):
            if not future.done():
                continue
            self.active_jobs.pop(request_id, None)
            duration_s = perf_counter() - started_at
            try:
                prepared_state = future.result()
            except Exception as exc:
                self.failed.emit(request_id, folder, str(exc), duration_s)
                continue
            payload = PatientActivationPreparationPayload(
                prepared_review_cache_state=prepared_state,
                cache_loaded=prepared_state is not None,
                cache_load_duration=duration_s if prepared_state is not None else None,
                used_preloaded_review_cache=prepared_state is not None,
            )
            self.finished.emit(request_id, folder, payload, duration_s)
        if not self.active_jobs:
            self.poll_timer.stop()


def load_patient_scan_and_discovery(folder: str) -> Tuple[CTVolume, PatientFileDiscovery]:
    return load_ct_series_and_discover_patient_files(folder)


def resample_dose_to_ct_volume(ct: CTVolume, dose: DoseVolume) -> np.ndarray:
    return np.stack(
        [sample_dose_to_ct_slice(ct, dose, slice_index) for slice_index in range(ct.volume_hu.shape[0])],
        axis=0,
    )


def _preload_structure_visible(rtstruct: Optional[RTStructData], idx: int) -> bool:
    if rtstruct is None or idx >= len(rtstruct.structures):
        return False
    return normalize_structure_name(rtstruct.structures[idx].name).startswith("PTV")


def _get_default_initial_slice_index(ct: CTVolume, rtstruct: Optional[RTStructData]) -> int:
    default_index = int(ct.volume_hu.shape[0] // 2)
    if rtstruct is None:
        return default_index

    indices: List[int] = []
    for structure in rtstruct.structures:
        if not normalize_structure_name(structure.name).startswith("PTV"):
            continue
        indices.extend(int(index) for index in structure.points_rc_by_slice.keys())
    if not indices:
        return default_index

    structure_start = min(indices)
    structure_end = max(indices)
    z_positions = ct.z_positions_mm
    start_pos = float(z_positions[structure_start] - 10.0)
    end_pos = float(z_positions[structure_end] + 10.0)
    start_idx = int(np.searchsorted(z_positions, start_pos, side="left"))
    end_idx = int(np.searchsorted(z_positions, end_pos, side="right") - 1)
    start_idx = max(0, min(start_idx, len(z_positions) - 1))
    end_idx = max(0, min(end_idx, len(z_positions) - 1))
    return int((start_idx + end_idx) // 2)


def _get_default_initial_slice_range(ct: CTVolume, rtstruct: Optional[RTStructData]) -> Tuple[int, int]:
    default_index = _get_default_initial_slice_index(ct, rtstruct)
    if rtstruct is None:
        return default_index, default_index

    indices: List[int] = []
    for structure in rtstruct.structures:
        if not normalize_structure_name(structure.name).startswith("PTV"):
            continue
        indices.extend(int(index) for index in structure.points_rc_by_slice.keys())
    if not indices:
        return default_index, default_index

    structure_start = min(indices)
    structure_end = max(indices)
    z_positions = ct.z_positions_mm
    start_pos = float(z_positions[structure_start] - 10.0)
    end_pos = float(z_positions[structure_end] + 10.0)
    start_idx = int(np.searchsorted(z_positions, start_pos, side="left"))
    end_idx = int(np.searchsorted(z_positions, end_pos, side="right") - 1)
    start_idx = max(0, min(start_idx, len(z_positions) - 1))
    end_idx = max(0, min(end_idx, len(z_positions) - 1))
    return start_idx, end_idx


def _sample_cine_indices(start_idx: int, end_idx: int, *, max_unique_frames: int = 12) -> List[int]:
    if end_idx <= start_idx:
        return [start_idx]
    span = end_idx - start_idx + 1
    if span <= max_unique_frames:
        indices = list(range(start_idx, end_idx + 1))
    else:
        indices = [
            int(round(value))
            for value in np.linspace(start_idx, end_idx, num=max_unique_frames)
        ]
        indices = list(dict.fromkeys(indices))
    if len(indices) > 1:
        return indices + indices[-2:0:-1]
    return indices


def _get_axial_cine_interval_ms(
    cine_indices: Sequence[int],
    spacing_z_mm: float,
    *,
    target_speed_mm_per_s: float = 7.5,
) -> int:
    if target_speed_mm_per_s <= 0.0:
        return 120
    if len(cine_indices) <= 1:
        step_mm = max(float(spacing_z_mm), 0.1)
    else:
        step_sizes_mm = [
            abs(int(cine_indices[idx + 1]) - int(cine_indices[idx])) * float(spacing_z_mm)
            for idx in range(len(cine_indices) - 1)
            if cine_indices[idx + 1] != cine_indices[idx]
        ]
        step_mm = max(float(np.mean(step_sizes_mm)), 0.1) if step_sizes_mm else max(float(spacing_z_mm), 0.1)
    interval_ms = int(round((step_mm / float(target_speed_mm_per_s)) * 1000.0))
    return max(60, interval_ms)


def build_axial_cine_plan(
    ct: CTVolume,
    rtstruct: Optional[RTStructData],
) -> Tuple[List[int], int]:
    cine_start_idx, cine_end_idx = _get_default_initial_slice_range(ct, rtstruct)
    cine_indices = _sample_cine_indices(cine_start_idx, cine_end_idx)
    cine_interval_ms = _get_axial_cine_interval_ms(cine_indices, float(ct.spacing_xyz_mm[2]))
    return cine_indices, cine_interval_ms


def _get_default_colorwash_min_dose_gy(
    rtstruct: Optional[RTStructData],
    plan_phases: Sequence[RTPlanPhase],
) -> float:
    available_phase_rx = [
        float(phase.prescription_dose_gy)
        for phase in plan_phases
        if phase.prescription_dose_gy > 0.0 and phase.dose_path
    ]
    ptv_dose_values: List[float] = []
    if rtstruct is not None:
        for structure in rtstruct.structures:
            normalized_name = normalize_structure_name(structure.name)
            if not normalized_name.startswith("PTV"):
                continue
            digits = "".join(ch for ch in normalized_name if ch.isdigit())
            if not digits:
                continue
            ptv_dose_values.append(float(int(digits)) / 100.0)

    if len(available_phase_rx) == 1:
        return available_phase_rx[0] * 0.95
    if len(available_phase_rx) > 1:
        if ptv_dose_values:
            return min(ptv_dose_values) * 0.95
        return min(available_phase_rx) * 0.95
    if ptv_dose_values:
        if len(ptv_dose_values) == 1:
            return ptv_dose_values[0] * 0.95
        return min(ptv_dose_values) * 0.95
    return 0.0


def build_precomputed_patient_view_state(
    *,
    ct: CTVolume,
    dose: Optional[DoseVolume],
    rtstruct: Optional[RTStructData],
    sampled_dose_volume_ct: Optional[np.ndarray],
    plan_phases: Sequence[RTPlanPhase],
) -> PrecomputedPatientViewState:
    slice_index = _get_default_initial_slice_index(ct, rtstruct)
    row_idx = int(ct.rows // 2)
    col_idx = int(ct.cols // 2)
    window_level = 40.0
    window_width = 400.0
    lo = window_level - window_width / 2.0
    hi = window_level + window_width / 2.0
    dose_alpha = 0.75
    dose_min_gy = _get_default_colorwash_min_dose_gy(rtstruct, plan_phases)
    dose_max_gy = float(np.nanmax(dose.dose_gy)) if dose is not None else 1.0
    if dose_max_gy <= dose_min_gy:
        dose_max_gy = dose_min_gy + max(dose_max_gy * 0.001, 1e-6)

    axial_render_state = build_axial_render_state(
        ct,
        dose,
        rtstruct,
        sampled_dose_volume_ct,
        slice_index,
        lo,
        hi,
        dose_alpha,
        dose_min_gy,
        dose_max_gy,
        lambda idx: _preload_structure_visible(rtstruct, idx),
    )
    orthogonal_render_state = build_orthogonal_render_state(
        ct,
        rtstruct,
        sampled_dose_volume_ct,
        row_idx,
        col_idx,
        lo,
        hi,
        dose_alpha,
        dose_min_gy,
        dose_max_gy,
        lambda idx: _preload_structure_visible(rtstruct, idx),
    )
    return PrecomputedPatientViewState(
        slice_index=slice_index,
        row_idx=row_idx,
        col_idx=col_idx,
        window_level=window_level,
        window_width=window_width,
        dose_alpha=dose_alpha,
        dose_min_gy=dose_min_gy,
        dose_max_gy=dose_max_gy,
        axial_render_state=axial_render_state,
        orthogonal_render_state=orthogonal_render_state,
    )


def prepare_patient_preload_payload(
    folder: str,
    *,
    array_cache_signature: Dict[str, str],
    progress_callback: Optional[Callable[[str], None]] = None,
    patient_plan_callback: Optional[Callable[[Optional[List[str]]], None]] = None,
    include_precomputed_view_state: bool = False,
) -> PatientPreloadPayload:
    timing_entries: List[Tuple[str, Optional[float]]] = []

    stage_start = perf_counter()
    ct, patient_discovery = load_patient_scan_and_discovery(folder)
    rtstruct_path = patient_discovery.rtstruct_path
    rtdose_paths = list(patient_discovery.rtdose_paths)
    rtplan_paths = list(patient_discovery.rtplan_paths)
    timing_entries.append(("CT scan/load + file discovery", perf_counter() - stage_start))
    if patient_plan_callback is not None:
        patient_plan_callback(patient_discovery.patient_plan_lines)

    stage_start = perf_counter()
    image_view_bounds = compute_image_view_bounds(ct)
    timing_entries.append(("Compute image bounds", perf_counter() - stage_start))

    rtstruct: Optional[RTStructData] = None
    if rtstruct_path:
        if progress_callback is not None:
            progress_callback("Loading structures")
        stage_start = perf_counter()
        rtstruct = load_rtstruct(rtstruct_path, ct)
        timing_entries.append(("Load RTSTRUCT", perf_counter() - stage_start))
    else:
        timing_entries.append(("Load RTSTRUCT", None))

    dose: Optional[DoseVolume] = None
    sampled_dose_volume_ct: Optional[np.ndarray] = None
    derived_array_cache_loaded = False
    derived_array_cache_data: Optional[DerivedArrayCacheData] = None
    review_cache_data: Optional[ReviewCacheFileData] = None
    precomputed_view_state: Optional[PrecomputedPatientViewState] = None
    if rtdose_paths:
        stage_start = perf_counter()
        dose = load_combined_rtdose(rtdose_paths)
        timing_entries.append(("Load/merge RTDOSE", perf_counter() - stage_start))

        derived_array_cache_path = get_derived_array_cache_path(get_dvh_cache_path(folder))
        stage_start = perf_counter()
        if derived_array_cache_path is not None and derived_array_cache_path.exists():
            derived_array_cache_data = load_derived_array_cache(
                derived_array_cache_path,
                ct=ct,
                rtstruct_path=rtstruct_path,
                rtdose_paths=rtdose_paths,
                array_cache_signature=array_cache_signature,
            )
        if derived_array_cache_data is not None:
            sampled_dose_volume_ct = derived_array_cache_data.sampled_dose_volume_ct
            derived_array_cache_loaded = sampled_dose_volume_ct is not None
        timing_entries.append(("Load derived array cache", perf_counter() - stage_start if derived_array_cache_loaded else None))

        if sampled_dose_volume_ct is None:
            if progress_callback is not None:
                progress_callback("Resampling dose")
            stage_start = perf_counter()
            sampled_dose_volume_ct = resample_dose_to_ct_volume(ct, dose)
            timing_entries.append(("Resample dose to CT grid", perf_counter() - stage_start))
        else:
            timing_entries.append(("Resample dose to CT grid", None))
    else:
        timing_entries.append(("Load/merge RTDOSE", None))
        timing_entries.append(("Load derived array cache", None))
        timing_entries.append(("Resample dose to CT grid", None))

    review_cache_path = get_dvh_cache_path(folder)
    if review_cache_path is not None and review_cache_path.exists():
        stage_start = perf_counter()
        review_cache_data = load_review_cache_file(review_cache_path)
        timing_entries.append(("Load saved review cache file", perf_counter() - stage_start if review_cache_data is not None else None))
    else:
        timing_entries.append(("Load saved review cache file", None))

    if include_precomputed_view_state:
        stage_start = perf_counter()
        precomputed_view_state = build_precomputed_patient_view_state(
            ct=ct,
            dose=dose,
            rtstruct=rtstruct,
            sampled_dose_volume_ct=sampled_dose_volume_ct,
            plan_phases=patient_discovery.plan_phases,
        )
        timing_entries.append(("Precompute initial view state", perf_counter() - stage_start))
    else:
        timing_entries.append(("Precompute initial view state", None))

    return PatientPreloadPayload(
        folder=folder,
        ct=ct,
        image_view_bounds=image_view_bounds,
        patient_plan_lines=patient_discovery.patient_plan_lines,
        plan_phases=list(patient_discovery.plan_phases),
        rtplan_paths=rtplan_paths,
        rtstruct_path=rtstruct_path,
        rtstruct=rtstruct,
        rtdose_paths=rtdose_paths,
        dose=dose,
        sampled_dose_volume_ct=sampled_dose_volume_ct,
        derived_array_cache_loaded=derived_array_cache_loaded,
        derived_array_cache_data=derived_array_cache_data,
        review_cache_data=review_cache_data,
        precomputed_view_state=precomputed_view_state,
        timing_entries=timing_entries,
    )


def get_review_cache_availability(
    *,
    dvh_can_start: bool,
    cache_path: Optional[Path],
    derived_array_cache_path: Optional[Path],
) -> ReviewCacheAvailability:
    cache_found = cache_path is not None and cache_path.exists()
    derived_sidecar_only = (
        dvh_can_start
        and not cache_found
        and derived_array_cache_path is not None
        and derived_array_cache_path.exists()
    )
    return ReviewCacheAvailability(
        dvh_can_start=dvh_can_start,
        cache_path=cache_path,
        cache_found=cache_found,
        derived_sidecar_only=derived_sidecar_only,
    )


def build_load_timing_report_text(
    *,
    folder: str,
    timing_entries: List[Tuple[str, Optional[float]]],
    constraints_path: Optional[str],
    constraints_sheet_name: Optional[str],
    rtstruct_path: Optional[str],
    rtdose_paths: List[str],
    ct: Optional[CTVolume],
    rtstruct: Optional[RTStructData],
    error_message: Optional[str] = None,
) -> str:
    lines = [
        "Peer Patient Load Timing Report",
        f"Generated: {datetime.now().isoformat(sep=' ', timespec='seconds')}",
        f"Patient folder: {folder}",
        "",
        "Summary",
        f"CT shape: {ct.volume_hu.shape if ct is not None else 'not loaded'}",
        f"Constraints file: {Path(constraints_path).name if constraints_path is not None else 'none'}",
        f"Constraints sheet: {constraints_sheet_name or 'none'}",
        f"RTSTRUCT file: {Path(rtstruct_path).name if rtstruct_path is not None else 'none'}",
        f"RTDOSE files: {len(rtdose_paths)}",
        f"Structures loaded: {len(rtstruct.structures) if rtstruct is not None else 0}",
        "",
        "Stage timings",
    ]

    for label, duration_s in timing_entries:
        if duration_s is None:
            if label == "Compute DVH (background)":
                lines.append(f"{label}: pending")
            else:
                lines.append(f"{label}: skipped")
        else:
            lines.append(f"{label}: {duration_s * 1000.0:.1f} ms ({duration_s:.3f} s)")

    if error_message:
        lines.extend(["", f"Error: {error_message}"])

    return "\n".join(lines) + "\n"
