from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CTVolume:
    volume_hu: np.ndarray
    slice_origins_xyz_mm: np.ndarray
    z_positions_mm: np.ndarray
    spacing_xyz_mm: np.ndarray
    image_orientation_patient: np.ndarray
    study_uid: str = ""
    frame_of_reference_uid: str = ""
    rows: int = 0
    cols: int = 0

    @property
    def origin_xyz_mm(self) -> np.ndarray:
        return self.slice_origins_xyz_mm[0]


@dataclass
class DoseVolume:
    dose_gy: np.ndarray
    slice_origins_xyz_mm: np.ndarray
    z_positions_mm: np.ndarray
    origin_xyz_mm: np.ndarray
    spacing_xyz_mm: np.ndarray
    image_orientation_patient: np.ndarray
    frame_of_reference_uid: str = ""
    dose_units: str = ""


@dataclass
class StructureSliceContours:
    name: str
    color_rgb: Tuple[int, int, int]
    points_rc_by_slice: Dict[int, List[np.ndarray]] = field(default_factory=dict)


@dataclass
class RTStructData:
    structures: List[StructureSliceContours]
    frame_of_reference_uid: str = ""


@dataclass
class RTPlanPhase:
    sop_instance_uid: str
    prescription_dose_gy: float
    fractions_planned: int
    dose_path: str = ""
    target_structure_name: str = ""
    plan_label: str = ""
    plan_name: str = ""


@dataclass
class PatientFileDiscovery:
    ct_paths: List[str]
    rtstruct_path: Optional[str]
    rtdose_paths: List[str]
    rtplan_paths: List[str]
    plan_phases: List[RTPlanPhase] = field(default_factory=list)
    patient_plan_lines: Optional[Tuple[str, ...]] = None


@dataclass
class StructureGoal:
    structure_name: str
    metric: str
    comparator: str
    value_text: str


@dataclass
class StructureGoalEvaluation:
    metric: str
    comparator: str
    goal_text: str
    actual_text: str
    passed: Optional[bool] = None


@dataclass
class DVHCurve:
    name: str
    color_rgb: Tuple[int, int, int]
    dose_bins_gy: np.ndarray
    volume_pct: np.ndarray
    voxel_count: int
    volume_cc: float
    mean_dose_gy: float
    max_dose_gy: float
    min_dose_gy: float
    volume_cc_axis: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    oversampling_factor: float = 1.0
    used_fractional_labelmap: bool = False
    metadata: Dict[str, float | str] = field(default_factory=dict)


@dataclass
class ImageViewBounds:
    axial_by_slice: Dict[int, Tuple[float, float, float, float]]
    sagittal: Optional[Tuple[float, float, float, float]] = None
    coronal: Optional[Tuple[float, float, float, float]] = None
