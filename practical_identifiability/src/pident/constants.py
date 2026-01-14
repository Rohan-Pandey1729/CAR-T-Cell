"""
Shared constants for the pident package.

Loads environment variables from .env file, including the path to the Monolix
command-line executable needed for parameter estimation.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

MONOLIX_PATH = os.getenv("MONOLIX_PATH")
if MONOLIX_PATH is None:
    raise RuntimeError(
        "MONOLIX_PATH environment variable is not set. "
        "Please set it in your .env file or as an environment variable."
    )
