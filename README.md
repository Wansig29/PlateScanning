# PSAU Gate Plate Scanner

An offline desktop app for the gate laptop. It watches the gate camera, reads the rear plate of each vehicle that drives in, and checks it against a local copy of the PSAU parking database. If the vehicle has an active violation, it alerts the guard right away.

```
camera ──▶ MOG2 motion gate ──▶ keep sharpest frames ──▶ plate localization ──▶ EasyOCR
                                   (Laplacian variance)     (classical + OCR detector)
                                                                                   │
  UI (live feed, logs, identity dashboard, captured plate) ◀── local SQLite lookup ◀┘
                                                                   ▲
                               native-app REST API ──(sync every 4 h / Sync Now)──┘
```

## Quick start (development)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q
```

To try the app with no backend and no camera:

```powershell
.\.venv\Scripts\python tools\seed_demo.py                       # demo vehicles + violations
.\.venv\Scripts\python tools\make_test_video.py gate.mp4 NBC1234 ABC1234 XYZ789 QWE4567
.\.venv\Scripts\python -m platescanner --source gate.mp4        # choose "Continue offline"
```

The demo plates show every result type:
- `NBC1234` has one active violation.
- `ABC1234` has two active violations, so the pager appears.
- `XYZ789` is registered with no active violation.
- `QWE4567` is not registered.

## Configuration

On first run, the app writes `config.json` to `%LOCALAPPDATA%\PlateScanner\`. You can point it somewhere else with the `PLATESCANNER_HOME` environment variable. The same folder holds the database, the downloaded photos, the plate captures and the logs.

| Setting | Meaning |
|---|---|
| `api.base_url` | Railway URL of the native-app API |
| `api.login_path`, `vehicles_path`, `violations_path` | Endpoint paths (see *Backend requirements*) |
| `camera.source` | `"0"` for the first USB camera, an `rtsp://…` URL for an IP camera, or a video file |
| `camera.roi` | `[x, y, w, h]` fractions of the frame to watch, e.g. `[0.2, 0.4, 0.6, 0.6]`, so trees or passers-by outside the lane don't trigger scans |
| `motion.min_area_ratio` | How much of the ROI must move to count as a vehicle |
| `ocr.plate_layouts` | Accepted plate formats (`L` = letter, `D` = digit) |
| `scan.plate_cooldown_seconds` | Ignore repeat reads of the same plate while it's still at the gate |
| `scan.fuzzy_match` | Accept a read that's one character off, if exactly one plate matches. The UI flags these as *approximate* |
| `sync.interval_hours` / `full_resync_hours` | Delta sync every 4 h; a full re-download every 24 h to drop records deleted online |

Keyboard: **F11** toggles full screen. `--fullscreen` starts the app in full screen.

## How it works

- **Motion gating** (`vision/motion.py`): MOG2 runs on a 320 px copy of each frame. Frames are only kept while a large enough area is moving, so a static scene costs almost nothing.
- **Best frame** (`EventTracker`): during each motion event the app keeps the 3 sharpest frames, scored by Laplacian variance over the moving region. When the vehicle stops at the gate or leaves, those frames go to OCR, sharpest first.
- **Plate localization and OCR** (`vision/plate.py`): a classical detector proposes plate-shaped regions (a bright rectangle with dark characters, or a dense text blob). EasyOCR reads each one. If none reads as a valid plate, EasyOCR's own text detector scans the whole vehicle. Reads are corrected to a known plate layout, e.g. `A8C1234` becomes `ABC1234`. Two-row motorcycle plates are also handled.
- **Lookup** (`db.py`): plates are matched on a confusion-folded key (O/0, I/1, B/8, S/5…), so a slightly misread plate still finds its record. Only local SQLite is queried, never the API.
- **Early reads**: while a vehicle is still moving, the best frame so far is OCR'd about every 0.8 s. Only the fast plate-crop path runs here, and only when a sharper frame has arrived. This lets a violator be flagged about 1 s after stopping instead of after the motion ends (`motion.early_ocr_seconds`).
- **Alert**: a violation turns the dashboard banner red and makes it flash, plays a beep, flashes the taskbar, and auto-expands the dashboard if it was collapsed. A labelled box with the OCR confidence % is also drawn on the live feed. The violation **stays on the dashboard until the guard clicks Acknowledge**, or for `scan.violation_hold_seconds`. Clear vehicles scanned meanwhile still go to Logs and Captured Plate.
- **Snapshots, not video**: no video is recorded. Each motion event saves one JPEG of its best frame, plus the plate crop, to `captures\YYYY-MM-DD\`. Events where no plate could be read are logged too; click the row to see the snapshot. The status bar shows what text OCR saw.
- **Chat-style feeds**: Logs and Captured Plate add new entries at the bottom and auto-scroll. Scrolling up pauses this and shows a "▼ N new scans" button.
- **Slow mode** (top bar): shows at most one scan per 2–10 s and queues the rest, so each is readable during busy periods. Violations always skip the queue.

### Multiple violations per vehicle

The spec left this open. The app stores **all** active violations. The dashboard shows the newest one in the wireframe's fields. When there are more, it rotates through them every 4 s, and the ‹ › buttons page manually (pausing rotation for 15 s). The banner and log entry say how many there are.

## Backend requirements (native-app, psau-security repo)

The scanner never reads violations over the network during a scan. Every lookup is local, so the API is only called during sync. What sync needs is **bulk delta endpoints**, not a per-plate filter:

1. `GET /api/security/vehicles?updated_since=<ISO8601>&page=&per_page=`, returning full vehicle records: plate, owner name, contact, owner photo URL, `updated_at`.
2. `GET /api/security/violations?updated_since=<ISO8601>&page=&per_page=`, returning full violation records: `vehicle_id` and/or plate, type, status, suspension start/end or duration, evidence photo URLs, `updated_at`. Resolved violations must still be returned with their new status, so the laptop learns they're no longer active.
3. Login with the existing guard account. The endpoint should accept `{email, password, device_name}` and return a Sanctum token.

`platescanner/mapping.py` accepts common Laravel field names (`plate_number`, `owner.name`, `violation_type.name`, `evidence[].url`, …) and plain or paginated responses. Once the real resource shape is fixed, trim it to the actual keys. Photo URLs on the API host are downloaded with the bearer token. Rate limiting matters less than first thought: the scanner makes at most a few paginated requests every 4 hours, plus manual syncs.

## Packaging

```powershell
.\build.ps1
```

This fetches the EasyOCR models into `models\` if needed (the gate laptop is offline, so the models are bundled), runs the tests, and builds `dist\PlateScanner\PlateScanner.exe`. Copy the whole `dist\PlateScanner` folder to the gate laptop.

This is a one-folder build rather than a single `.exe` on purpose. PyTorch makes the bundle several hundred MB, and a single-file build would unpack all of it to a temp folder on every launch (30 s+ start-up).

## Security notes

- The guard's API token is encrypted with Windows DPAPI (`session.bin`), so it's only readable by the same Windows user on that laptop.
- An expired token (HTTP 401) prompts for sign-in again. Scanning continues on the local database the whole time.
