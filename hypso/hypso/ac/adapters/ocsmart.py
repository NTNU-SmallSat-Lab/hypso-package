"""OC-SMART adapter. Unlike Polymer/ACOLITE, OC-SMART has no importable
Python API - it's invoked as a bare `python OCSMART.py` subprocess reading
every run parameter (including its own input/output directories) from a
fixed-path global config file (OCSMART_Input.txt) in its own installation
directory. run_correction's design (capture-local staging, config save/
restore even on failure, output routed into the capture's own directory,
streamed console output) is folded in verbatim from
hypso-processing-pipeline's own field-tested ac_runners_hypso.
run_ocsmart_correction - built there specifically to replace this package's
earlier, simpler version (bare "python3", shared install-dir staging, output
left in OC-SMART's own L2/) after real problems with that version were found
in production use (a concurrent-run collision on the shared staging dir) and
a real naming bug (see HYPSO_PREFIX below) was found independently by both
that pipeline's own debugging and while building this adapter.

stage_input/run_correction (two separate calls) are MERGED into one
run_correction here - unlike the earlier split, this was never actually two
independent steps: the OCSMART_Input.txt save/restore has to wrap staging and
the subprocess call together to correctly restore on any failure in between.
Confirmed zero external callers of the old two-call split anywhere (this
repo, hypso-processing-pipeline, the original hypso-package) before merging."""
import shutil
import subprocess
from pathlib import Path

import numpy as np

from hypso.load import load_ocsmart_h5

from .base import ACAdapter, ACRunError, get_inferred_wavelength_band_map


class OCSMARTAdapter(ACAdapter):

    key = "ocsmart"

    # OC-SMART's own src/sensorinfo.py autodetect() only recognizes a HYPSO
    # input file by this exact, satellite-agnostic prefix ("HYPSO_HSI", no
    # satellite digit) - it doesn't distinguish HYPSO-1 vs HYPSO-2, both are
    # loaded via the single auxdata/sensorinfo/HYPSO_HSI.txt coefficient
    # file. satobj.sensor (e.g. "hypso2_hsi") does NOT match this - staging
    # under that name instead produces OC-SMART's "Unable to detect sensor"
    # warning with no output, silently (exit code 0, no exception). This is
    # a hardcoded fact about this specific OC-SMART release, not something
    # to source from config.
    HYPSO_PREFIX = "HYPSO_HSI"

    def output_path(self, satobj) -> Path:
        """Where the OC-SMART HDF5 output ends up, in the capture's own
        directory (not OC-SMART's shared install-dir L2/) - matching every
        other AC method's output convention. Callable before running to
        locate an already-produced output without re-running OC-SMART."""
        return Path(satobj.capture_dir, f"{self.HYPSO_PREFIX}_{satobj.capture_name}-l1d_L2_OCSMART.h5")

    def run_correction(self, satobj,
                       l2_prod: str = "Lt,Lr,Lrc,rrs,chl,tg_sol,tg_sen,Lwp",
                       solz_limit: float = 70.0, senz_limit: float = 70.0,
                       skip_existing: bool = True,
                       python_path: str = None):
        """
        Run OC-SMART on this capture's L1D file. Stages the input into
        capture_dir/ocsmart_staging/l1b/ (NOT OC-SMART's own shared
        ocsmart_dir/L1B/ - a real concurrent-run collision was observed
        there in production, two runs' staged files landing in the same
        shared directory at once) under the filename convention OC-SMART's
        own sensor autodetection requires (HYPSO_PREFIX), and writes output
        directly into capture_dir (see output_path()).

        OC-SMART has no CLI arguments or importable Python API - every run
        parameter (including its own input/output paths) comes from a
        fixed-path global config file (OCSMART_Input.txt) in its own
        installation directory - unavoidably shared, global state for this
        OC-SMART install regardless of where the actual data files live.
        This overwrites that file for the duration of this call and
        restores its previous content afterward, even on failure. WARNING:
        two concurrent OC-SMART runs against the SAME installation would
        still race on writing/reading this one shared file, even though the
        actual input/output data no longer collides (each run stages into
        its own capture-specific directory).

        python_path: interpreter OCSMART.py itself runs under - OC-SMART
            ships its own pinned (and Python-version-specific)
            environment.yml, typically incompatible with whatever
            environment this call itself runs under. Defaults to bare
            "python3" on PATH if not given. Unlike Polymer/ACOLITE's
            python_path (which selects the interpreter running THIS
            adapter's own subprocess, defaulting to sys.executable), this
            one only ever selects the NESTED OCSMART.py subprocess's
            interpreter - there is no meaningful "same as us" default here,
            since OC-SMART is never imported by this adapter's own process
            either way.
        skip_existing: if True (default) and output_path(satobj) already
            exists, OC-SMART is not re-run.
        l2_prod, solz_limit, senz_limit: OCSMART_Input.txt's own fields -
            defaulted to values already validated against real HYPSO
            captures.

        Raises hypso.ac.adapters.base.ACRunError on failure.
        """
        ocsmart_dir = Path(satobj.ocsmart_dir).absolute()
        capture_dir = Path(satobj.capture_dir)
        dest_file = self.output_path(satobj)

        if skip_existing and dest_file.is_file():
            print(f"[INFO] OC-SMART output already exists at {dest_file}. Skipping.")
            return dest_file

        staged_name = f"{self.HYPSO_PREFIX}_{satobj.capture_name}-l1d.nc"
        l1b_staging_dir = capture_dir / "ocsmart_staging" / "l1b"
        l1b_staging_dir.mkdir(parents=True, exist_ok=True)
        capture_dir.mkdir(parents=True, exist_ok=True)

        staged_file = l1b_staging_dir / staged_name
        shutil.copy2(satobj.l1d_nc_file, staged_file)
        print(f"[INFO] Staged OC-SMART input file to {staged_file}")

        input_txt_path = ocsmart_dir / "OCSMART_Input.txt"
        original_input_txt = input_txt_path.read_text() if input_txt_path.is_file() else None

        # Absolute paths are safe here (unlike ocsmart_dir's own path, which
        # may contain literal dots, e.g. "OCSMART_Linux_v2.6.3", and gets
        # mangled by OCSMART.py's blind var.replace(".", OCSMART_script_dir)
        # on every "*path" value) because HYPSO capture directory names
        # (site_YYYY-MM-DDTHH-MM-SSZ) never contain a literal ".".
        input_txt_path.write_text(
            f"l1b_path = {l1b_staging_dir}/\n"
            f"l2_path = {capture_dir}/\n"
            f"l2_prod = {l2_prod}\n"
            f"solz_limit = {solz_limit}\n"
            f"senz_limit = {senz_limit}\n"
        )

        interpreter = str(python_path) if python_path else "python3"
        print(f"[INFO] Running OC-SMART atmospheric correction as a subprocess (interpreter: {interpreter})")

        stdout_lines = []
        try:
            # OC-SMART writes its own console output straight to the real
            # stdout/stderr file descriptors it inherits - subprocess.run()'s
            # default (stdout=None) does NOT go through Python's sys.stdout,
            # so a caller that later wraps this in something like a
            # tee-to-file logger would never see it there, and it could
            # interleave unpredictably with our own print()/logger output on
            # screen. Piping and re-printing line-by-line here routes it
            # back through Python's sys.stdout instead.
            process = subprocess.Popen(
                [interpreter, "OCSMART.py"], cwd=ocsmart_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            for line in process.stdout:
                print(line, end="")
                stdout_lines.append(line)
            return_code = process.wait()
            if return_code != 0:
                raise ACRunError(
                    tool="ocsmart", returncode=return_code,
                    stdout="".join(stdout_lines), stderr="",
                )
        finally:
            staged_file.unlink(missing_ok=True)
            # Best-effort only - a failure here must not mask a real
            # exception already propagating out of this finally block.
            try:
                l1b_staging_dir.rmdir()
                l1b_staging_dir.parent.rmdir()
            except OSError:
                pass
            if original_input_txt is not None:
                input_txt_path.write_text(original_input_txt)

        if not dest_file.is_file():
            raise ACRunError(
                tool="ocsmart", returncode=0,
                stdout="".join(stdout_lines), stderr="",
                error_type="FileNotFoundError",
                message=(f"No OC-SMART output found at {dest_file} after a "
                        f"zero-exit run - OC-SMART likely failed to "
                        f"autodetect the sensor from the staged input "
                        f"filename; check the console output above."),
            )

        print(f"[INFO] OC-SMART atmospheric correction complete: {dest_file}")
        return dest_file

    def open_output(self, satobj, h5_file_path: Path = None):
        """
        Open and read OC-SMART atmospheric correction HDF5 output files. The remote sensing reflectance (Rrs) dataset is written to the satobj's 'l2a_cube' dictionary.

        :param h5_file_path: Path to the OC-SMART HDF5 file (optional - defaults to output_path(satobj))

        :return: "datasets" Dictionary containing 2D and 3D datasets read from the HDF5 and stored as xarray DataArrays.
        """

        if h5_file_path is not None:
            h5_file_path = Path(h5_file_path).absolute()
        else:
            h5_file_path = self.output_path(satobj)

        if h5_file_path.is_file():
            print("[INFO] Opening OC-SMART output file " + str(h5_file_path))
            datasets = load_ocsmart_h5(h5_file_path = h5_file_path)

        else:
            print("[ERROR] OC-SMART output file " + str(h5_file_path) + " does not exist.")
            return None

        try:
            key = "Rrs"
            inferred_wavelengths = datasets[key].band.to_numpy()

            # Map inferred OC-SMART wavelengths to HYPSO wavelengths
            wl_band_map = get_inferred_wavelength_band_map(satobj, inferred_wavelengths=inferred_wavelengths)

            # Create empty cube with standard HYPSO cube dims
            shape = (satobj.spatial_dimensions[0], satobj.spatial_dimensions[1], satobj.bands)
            cube = np.full(shape=shape, fill_value=np.nan)
            cube[:,:,wl_band_map] = datasets[key]

            satobj.l2a_cube["ocsmart"] = cube
            satobj.l2a_cube["ocsmart"].attrs['l2_variable_name'] = key

        except Exception as ex:
            print("[ERROR] Unable to load OC-SMART L2 Rrs dataset.")

        return datasets
