"""Make the repository root importable during tests.

`python -m pytest` puts the working directory on sys.path, but a bare `pytest`
does not -- it only adds each test file's own directory. Without this the suite
passes locally and fails in CI with `ModuleNotFoundError: No module named 'pve'`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
