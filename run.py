"""PyInstaller entry point.

The packaged app has no console, so a startup failure (a missing module, a bad config) would just make
the window vanish. Anything that goes wrong is written to crash.log and shown in a message box instead.
"""
import os
import sys
import traceback
from pathlib import Path


def _report_crash() -> None:
    text = traceback.format_exc()
    base = os.environ.get("PLATESCANNER_HOME") or os.path.join(
        os.environ.get("LOCALAPPDATA") or str(Path.home()), "PlateScanner")
    path = Path(base) / "crash.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None, f"PlateScanner could not start.\n\nDetails were saved to:\n{path}\n\n{text[-900:]}",
            "PSAU Gate Plate Scanner", 0x10)
    else:
        print(text, file=sys.stderr)


if __name__ == "__main__":
    try:
        from platescanner.app import main
        main()
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        _report_crash()
        sys.exit(1)
