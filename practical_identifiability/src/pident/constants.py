import os

from dotenv import load_dotenv

load_dotenv()

MONOLIX_PATH = os.getenv("MONOLIX_PATH")
if MONOLIX_PATH is None:
    raise RuntimeError(
        "MONOLIX_PATH environment variable is not set. "
        "Please set it in your .env file or as an environment variable."
    )
