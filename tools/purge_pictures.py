"""Delete saved pictures of scans that are not violations (the log rows are kept).

    python tools\purge_pictures.py              # shows what would be deleted, deletes nothing
    python tools\purge_pictures.py --yes        # deletes it
    python tools\purge_pictures.py --keep violation no_plate --yes

By default it keeps the results listed in scan.save_pictures_for (["violation"]). Close the
scanner first.

Delete test scans entirely (the log rows too, violations included), by date:

    python tools\purge_pictures.py --delete-before 2026-10-01          # shows what would go
    python tools\purge_pictures.py --delete-before 2026-10-01 --yes    # deletes scans dated before Oct 1

Clear one day's scans (today, or a given day), with their pictures:

    python tools\purge_pictures.py --delete-today                      # shows what would go
    python tools\purge_pictures.py --delete-today --yes                # deletes today's scans
    python tools\purge_pictures.py --delete-on 2026-10-06 --yes        # deletes Oct 6's scans
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from platescanner import db, purge  # noqa: E402
from platescanner.config import load_config  # noqa: E402


def delete_scans(conn, roots, label: str, plan_fn, value: str, yes: bool) -> None:
    try:
        plan = plan_fn(conn, roots, value)
    except ValueError:
        sys.exit(f"'{value}' is not a date: use YYYY-MM-DD, e.g. 2026-10-01")
    breakdown = ", ".join(f"{n} {db.CAPTURE_FOLDERS.get(r, r)}" for r, n in sorted(plan.by_result.items()))
    print(f"Scans {label}: {len(plan.scan_ids)}" + (f" ({breakdown})" if breakdown else ""))
    print(f"{len(plan.files)} pictures ({plan.bytes / 1_000_000:.1f} MB) go with them. "
          "The scan records are deleted too, violations included.")
    if not yes:
        print("Nothing was deleted. Run again with --yes to delete.")
        return
    scans, pictures = purge.run_delete_before(conn, plan, roots)
    print(f"Deleted {scans} scans and {pictures} pictures.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", nargs="*", help="results whose pictures are kept (default: scan.save_pictures_for)")
    ap.add_argument("--delete-before", metavar="YYYY-MM-DD",
                    help="delete every scan dated before this day, with its pictures (instead of purging pictures)")
    ap.add_argument("--delete-on", metavar="YYYY-MM-DD",
                    help="delete every scan dated this day, with its pictures")
    ap.add_argument("--delete-today", action="store_true",
                    help="delete every scan dated today, with its pictures")
    ap.add_argument("--yes", action="store_true", help="really delete (without it, only report)")
    args = ap.parse_args()

    cfg = load_config()
    keep = set(args.keep if args.keep is not None else cfg.scan.save_pictures_for)
    roots = [cfg.captures_dir, cfg.archive_path]
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    if args.delete_before:
        delete_scans(conn, roots, f"dated before {args.delete_before}", purge.plan_delete_before,
                     args.delete_before, args.yes)
        return
    day = date.today().isoformat() if args.delete_today else args.delete_on
    if day:
        delete_scans(conn, roots, f"dated {day}", purge.plan_delete_day, day, args.yes)
        return
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
