from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence


def _configure_low_priority_worker() -> None:
    thread_limit_env = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, value in thread_limit_env.items():
        os.environ.setdefault(key, value)

    try:
        os.nice(15)
    except Exception:
        pass

    try:
        import resource  # type: ignore

        resource.setpriority(resource.PRIO_PROCESS, 0, 15)
    except Exception:
        pass


def prepare_activation_review_cache_state(
    review_cache_data: object,
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
) -> object:
    _configure_low_priority_worker()

    from peer_cache import (
        default_is_base_listable_structure_name,
        prepare_review_cache_state,
    )

    return prepare_review_cache_state(
        review_cache_data,
        expected_structure_names=list(expected_structure_names),
        available_constraint_sheet_names=list(available_constraint_sheet_names),
        no_constraints_sheet_label=no_constraints_sheet_label,
        constraints_sheet_name=constraints_sheet_name,
        structure_filter_csv_path=structure_filter_csv_path,
        constraint_script_xml_path=constraint_script_xml_path,
        script_constraints_label=script_constraints_label,
        rtstruct_path=rtstruct_path,
        rtdose_paths=list(rtdose_paths),
        rtplan_paths=list(rtplan_paths),
        dvh_mode=dvh_mode,
        dvh_method_signature=dvh_method_signature,
        target_method_signature=dict(target_method_signature),
        has_ct=has_ct,
        has_dose=has_dose,
        is_base_listable_structure_name=default_is_base_listable_structure_name,
    )
