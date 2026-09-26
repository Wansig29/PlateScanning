import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Keep tests out of the real %LOCALAPPDATA%\PlateScanner.
os.environ["PLATESCANNER_HOME"] = tempfile.mkdtemp(prefix="platescanner-test-")
