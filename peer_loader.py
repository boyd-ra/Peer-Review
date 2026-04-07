from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from peer_helpers import sample_dose_to_ct_slice
from peer_io import load_ct_series_and_discover_patient_files
from peer_models import CTVolume, DoseVolume, PatientFileDiscovery, RTStructData


def load_patient_scan_and_discovery(folder: str) -> Tuple[CTVolume, PatientFileDiscovery]:
    return load_ct_series_and_discover_patient_files(folder)


def resample_dose_to_ct_volume(ct: CTVolume, dose: DoseVolume) -> np.ndarray:
    return np.stack(
        [sample_dose_to_ct_slice(ct, dose, slice_index) for slice_index in range(ct.volume_hu.shape[0])],
        axis=0,
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
