import subprocess
from pathlib import Path

from pident.constants import MONOLIX_PATH


def run_monolix_estimation(
    mlxtran_path: str | Path,
    output_path: str | Path,
    mode: str = "complete",
    n_threads: int = 1,
    wait: bool = True,
) -> subprocess.Popen:
    """
    Run a Monolix estimation with the given configuration file and options.

    Args:
        mlxtran_path: Path to the .mlxtran configuration file (will be converted to absolute if relative)
        output_path: Path where Monolix should write output (corresponds to -o flag)
        mode: Monolix execution mode (default: "complete")
        n_threads: Number of threads for Monolix to use (default: 1)
        wait: If True (default), block until the Monolix process finishes.
              If False, return immediately and let the caller manage the process.

    Returns:
        The subprocess.Popen object. If wait=True, the process will have already finished.

    Raises:
        RuntimeError: If MONOLIX_PATH is not set (from constants.py)
    """
    mlxtran_path = Path(mlxtran_path).resolve()
    output_path = Path(output_path).resolve()

    if not mlxtran_path.exists():
        raise FileNotFoundError(f"Monolix configuration file not found: {mlxtran_path}")

    process = subprocess.Popen(
        [
            str(MONOLIX_PATH),
            "--no-gui",
            "--mode",
            mode,
            "--thread",
            str(n_threads),
            "-p",
            str(mlxtran_path),
            "-o",
            str(output_path),
        ]
    )

    if wait:
        process.wait()

    return process
