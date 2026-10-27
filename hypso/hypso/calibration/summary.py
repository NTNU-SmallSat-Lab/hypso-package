"""One summary of a calibration run, logged at INFO by both calibration
paths (hypso.calibration.pipeline.run_calibration and
hypso.processing.l1a_to_l1b). The same details are recorded in the product's
metadata/processing/l1b provenance."""
from pathlib import Path

_CORRECTIONS = ("radiometric", "keystone", "smile", "destriping", "spectral")


def format_calibration_summary(sat_id: str, capture_type: str, files: dict, applied, selection: dict = None) -> str:
    """`files`: {correction: path or None}; `applied`: corrections actually
    run; `selection`: hypso2_calibration's resolve_coefficients() result
    (None for sensors/sets without versioned coefficients)."""
    if selection:
        header = (f"Calibration summary: {sat_id}, capture type {capture_type}, coeff_type "
                  f"{selection['coeff_type']}, release {selection['release']}")
    else:
        header = f"Calibration summary: {sat_id}, capture type {capture_type}"
    lines = [header]
    for correction in _CORRECTIONS:
        path = files.get(correction)
        version = (selection or {}).get(correction)
        if correction in applied:
            state = "applied"
        elif correction == "keystone" and path is None:
            state = "off (opt-in: keystone=True)"
        elif path is None:
            state = "skipped (no coefficients)"
        else:
            state = "skipped"
        detail = f" {version}" if version else ""
        name = f"  {Path(str(path)).name}" if path is not None else ""
        lines.append(f"  {correction:<12} {state}{detail}{name}")
    return "\n".join(lines)
