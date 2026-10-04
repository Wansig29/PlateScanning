"""Delete saved pictures of scans that are not violations (the log rows are kept).

    python tools\purge_pictures.py              # shows what would be deleted, deletes nothing
    python tools\purge_pictures.py --yes        # deletes it
    python tools\purge_pictures.py --keep violation no_plate --yes

By default it keeps the results listed in scan.save_pictures_for (["violation"]). Close the
scanner first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from platescanner import db, purge  # noqa: E402
from platescanner.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", nargs="*", help="results whose pictures are kept (default: scan.save_pictures_for)")
    ap.add_argument("--yes", action="store_true", help="really delete (without it, only report)")
    args = ap.parse_args()

    cfg = load_config()
    keep = set(args.keep if args.keep is not None else cfg.scan.save_pictures_for)
    roots = [cfg.captures_dir, cfg.archive_path]
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    plan = purge.plan_purge(conn, roots, keep)
    print(f"Keeping pictures of: {', '.join(sorted(keep)) or 'nothing'}")
    print(f"Folders searched: {', '.join(str(r) for r in roots)}")
    print(f"{len(plan.files)} pictures ({plan.bytes / 1_000_000:.1f} MB) would be deleted; "
          f"{len(plan.scan_ids)} scan records lose their picture links (the records stay).")
    if not args.yes:
        print("Nothing was deleted. Run again with --yes to delete.")
        return
    n = purge.run_purge(conn, plan, roots)
    print(f"Deleted {n} pictures.")


if __name__ == "__main__":
    main()
