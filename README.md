# PSAU Gate Plate Scanner

An offline desktop app for the gate laptop. It watches the gate camera, reads the plate of **every vehicle in view (several at once, without them having to stop)** and checks each against a local copy of the PSAU parking database. If a vehicle has an active violation, the guard is alerted within about half a second of its plate coming into view.

```
camera ──▶ newest frame ──▶ plate detector ──▶ tracker ──────▶ plate OCR ×2 ──▶ vote over the
  │        (older frames     (neural YOLOv9     (one ID per      (two models,      vehicle's reads
  │         dropped)          + classical)       vehicle)         merged)                │
  ▼                                                                                      ▼
MOG2 motion gate (skips work when idle)                                  local SQLite lookup
                                                                                         │
  UI (live feed with labelled vehicles, logs, identity dashboard, captured plates) ◀─────┘
                                                                         ▲
                        native-app REST API ──(sync every 4 h / Sync Now)┘ (into SQLite)
```

## Quick start (development)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python tools\fetch_models.py                   # plate models into models\alpr (13 MB)
.\.venv\Scripts\python -m pytest -q
```

To try the app with no backend and no camera:

```powershell
.\.venv\Scripts\python tools\seed_demo.py                       # demo vehicles + violations
.\.venv\Scripts\python tools\make_test_video.py gate.mp4 NBC1234 ABC1234 XYZ789 QWE4567 --no-stop --blur --gap 0.6 --lanes 2
.\.venv\Scripts\python -m platescanner --source gate.mp4        # choose "Continue offline"
```

`make_test_video.py` options: `--no-stop` (drive through instead of stopping), `--speed` (px/frame; 18 ≈ 6.5 km/h, 45 ≈ 16 km/h), `--gap` (seconds between cars, so several are in view), `--lanes 2`, `--blur` / `--shutter` (motion blur from speed × exposure time).

The demo plates show every result type:
- `NBC1234` has one active violation.
- `ABC1234` has two active violations, so the pager appears.
- `XYZ789` is registered with no active violation.
- `QWE4567` is not registered.

## Configuration

On first run, the app writes `config.json` to `%LOCALAPPDATA%\PlateScanner\`. You can point it somewhere else with the `PLATESCANNER_HOME` environment variable. The same folder holds the database, the downloaded photos, the plate captures and the logs.

| Setting | Meaning |
|---|---|
| `api.base_url` | psau-security address (default `https://psau-security-production.up.railway.app`) |
| `api.login_path`, `vehicles_path`, `violations_path` | Endpoint paths (see *Connection to psau-security*) |
| `camera.source` | `"0"` for the first USB camera, an `rtsp://…` URL for an IP camera, or a video file |
| `camera.roi` | `[x, y, w, h]` fractions of the frame to watch, e.g. `[0.2, 0.4, 0.6, 0.6]`, so trees or passers-by outside the lane don't trigger scans |
| `motion.min_area_ratio` | How much of the ROI must move to count as a vehicle |
| `ocr.plate_layouts` | Accepted plate formats (`L` = letter, `D` = digit) |
| `ocr.detector_model` | Plate detector input size. `…-t-384-…` (default) is the best speed/recall trade-off on a CPU; `…-t-512-…` / `…-t-640-…` find smaller, farther plates but are slower |
| `ocr.ocr_model` | Plate OCR models, comma-separated. Each plate is read by all of them and the results are merged character by character |
| `ocr.read_confidence`, `report_confidence`, `alert_confidence`, `confirm_reads` | When a vehicle is reported (see *Deciding a plate*) |
| `ocr.classical_proposals` | Also try plate candidates from the classical finder (catches plates the neural detector misses, ~20 ms per frame) |
| `scan.plate_cooldown_seconds` | Ignore repeat reads of the same plate while it's still at the gate |
| `scan.fuzzy_match` | Accept a read that's one character off, if exactly one plate matches. The UI flags these as *approximate* |
| `scan.reminder_seconds` | Repeat the alarm this often while a violation is unacknowledged (0 = alert once only) |
| `scan.bring_to_front` | Bring the app to the front on every violation alert |
| `sync.interval_hours` / `full_resync_hours` | Delta sync every 3 h (nothing is written when there are no new vehicles or violations); a full re-download every 24 h to drop records deleted online |

Keyboard: **F11** toggles full screen. `--fullscreen` starts the app in full screen.

## How it works

- **Newest frame only** (`pipeline.py`): the camera thread hands the recognizer only the latest frame; older ones are dropped. Recognition therefore always works on what is at the gate *now* and never builds up a backlog in heavy traffic. Test videos are played the same way (in real time, dropping frames when behind), so they behave like a live camera.
- **Motion gate** (`vision/motion.py`): MOG2 on a 320 px copy of each frame. Only vehicle-like motion counts: one solid moving shape covering at least `motion.min_area_ratio` of the picture. People (shapes taller than `motion.max_height_ratio` × their width), leaves and rain (specks that are mostly empty space when grouped) and lighting changes (more than `motion.max_area_ratio` of the picture at once) are ignored. When there is no vehicle motion and no real plate is being followed, plate detection runs only about once a second, so an empty gate costs almost no CPU. That once-a-second look also catches a vehicle the motion rules missed.
- **Assembly line**: the next frame goes through the plate detector while the plates of the current frame are being read, and all plates in a frame are read in parallel. On an 8-core laptop this raised the scanned frames from about 14 to 16–18 per second.
- **Plate detection** (`vision/alpr.py`): a YOLOv9 plate detector (open-image-models, ONNX, MIT license) finds every plate in the frame. The classical contrast/edge finder (`vision/plate.py`) adds candidates the network misses. Wrong candidates are harmless: their text never fits a plate layout.
- **One track per vehicle** (`vision/tracker.py`): detections are matched frame to frame by overlap and predicted motion, so each vehicle keeps its ID (`#12`) while it crosses the picture. Several vehicles, side by side or bumper to bumper, are followed at once. Plates cut off by the edge of the picture are tracked but not read until they are fully in view.
- **Motion blur** (`ocr.deblur`): a plate whose sideways edges are much weaker than its up-and-down ones is blurred by the vehicle's motion. Its blur length is measured (cepstrum) and undone (Wiener deconvolution) before reading, because blurred plates are often read *confidently wrong*. On test videos at 16–25 km/h this took reads from 4/8 and 0/8 to 8/8 and 7/8. At 40–50 km/h the camera's shutter decides: 8/8 at 1/1000 s, 7/8 at 1/250 s, 5/8 at a typical webcam's 1/100 s.
- **Plate OCR** (fast-plate-ocr, ONNX, MIT license): each plate crop is read by two small OCR models that tend to make different mistakes. Their outputs are layout-corrected (`A8C1234` → `ABC1234`) and merged character by character. On the test set this pair reads more plates correctly than the single model ten times its size.
- **Deciding a plate**: all reads of one vehicle are combined character by character, weighted by confidence, so a frame that misreads one letter is outvoted by the others. A vehicle is reported as soon as the evidence is strong enough:
  - a **violation** on the first read that is at least `alert_confidence` (0.75) sure, or on `confirm_reads` (2) agreeing reads;
  - anything else once `confirm_reads` reads agree;
  - if the vehicle leaves first, its best guess is reported if it averages `report_confidence` (a violation only needs `read_confidence`, because missing a violator is worse than a doubtful alert). Otherwise it is logged as *plate not readable*, with a snapshot.

  After reporting, a vehicle is still re-read now and then. If the consensus clearly changes, a correction is logged.
- **Lookup** (`db.py`): plates are matched on a confusion-folded key (O/0, I/1, B/8, S/5…), so a slightly misread plate still finds its record. Only local SQLite is queried, never the API.
- **Live feed**: every tracked vehicle gets a box, a trail and a label (`#12 ABC 1234 98%`, or `#13 reading...` until decided). Between recognitions, boxes move with the vehicle's measured speed so they stay on the plate. While a **violator** is in view the rest of the picture is dimmed and the violator's whole vehicle gets a thick, blinking red frame, so it stands out from the cars around it.
- **Which vehicle is it?** (`vision/identify.py`): with several cars at the gate a plate number alone is hard to match to a moving car, so the dashboard leads with:
  - **This vehicle**: a photo of the vehicle itself, cut from the camera frame;
  - **Where it was**: the whole scene at that moment with only this vehicle highlighted (others outlined in grey);
  - its **colour**, its **position** in the picture (left / middle / right), how many **other vehicles** were in view, and its live status: **In view now**, or *Left the picture 8 s ago*;
  - the banner names the vehicle number (`VEHICLE #12`) that labels it on the live feed.

  Owner, violation and suspension follow in one compact card; the violation's evidence photos open from a link. The Logs and Captured Plates show the colour and position too (*Red, left side · Illegal parking*), and all of it is saved with the scan, so reopening a scan from the Logs shows the same pictures.
- **No violator goes unnoticed**: every violation alert stays open until a guard acknowledges it.
  - The alert itself: the dashboard banner turns red and flashes, an alarm sounds, the taskbar flashes, the dashboard expands if collapsed, and the app **brings itself to the front** if another window covers it (`scan.bring_to_front`; Windows may only flash the taskbar when another program has the focus).
  - While anything is unacknowledged: a **pulsing red frame** runs around the whole window (visible from across the booth), a red **"⚠ N unacknowledged violations"** counter sits in the top bar, and the whole alert **repeats every `scan.reminder_seconds`** (15 s) until someone responds.
  - The violator **stays on the dashboard** until acknowledged; however much traffic follows, other vehicles never replace it.
  - **Several violators at once**: each gets its own alarm (played one after another, never on top of each other). The dashboard shows the first and queues the rest (*Acknowledge violation · 2 more waiting*); each Acknowledge brings up the next. Opening an older scan from the Logs keeps the queue and offers *Back to violations*.
  - **Who acknowledged what, and when** is stored with the scan (the signed-in guard's name) and shown in the Logs (*✓ acknowledged by Juan at 10:05:12*, or a red *⚠ NOT ACKNOWLEDGED*).
  - It **survives a restart**: violations nobody acknowledged in the last `scan.unacknowledged_lookback_hours` (24 h) are raised again when the app starts (unless the violation was resolved online in the meantime).
- **Snapshots, not video**: no video is recorded. Each reported vehicle saves one JPEG of its best frame, plus the plate crop, to `captures\YYYY-MM-DD\`. Plates that were seen but never readable are logged too; click the row to see the snapshot.
- **Chat-style feeds**: Logs and Captured Plates add new entries at the bottom and auto-scroll. Scrolling up pauses this and shows a "▼ N new scans" button.
- **Slow mode** (Logs header): adds at most one log entry per 2–10 s and queues the rest, so each can be read during busy periods. It only paces the Logs: the dashboard, captured plates and alerts are always immediate, violations skip the queue, and every entry keeps the time its vehicle was actually scanned.

### Measuring speed and accuracy

```powershell
.\.venv\Scripts\python tools\make_test_video.py rush.mp4 NBC1234 ABC1234 XYZ789 QWE4567 DEF5678 GHJ2345 --no-stop --blur --speed 30 --gap 0.8 --lanes 2
.\.venv\Scripts\python tools\bench_live.py rush.mp4
.\.venv\Scripts\python tools\bench_live.py real_gate.mp4 --set camera.roi=[0.1,0.3,0.8,0.7]   # real footage: lists every read
```

`bench_live.py` plays a video in real time through the same threads the app uses and reports, for each vehicle, whether it was read, whether correctly, and how long after its plate appeared. Run it on footage from the actual gate camera before deploying, and use `--set section.name=value` to try settings.

### Multiple violations per vehicle

The spec left this open. The app stores **all** active violations. The dashboard shows the newest one in the wireframe's fields. When there are more, it rotates through them every 4 s, and the ‹ › buttons page manually (pausing rotation for 15 s). The banner and log entry say how many there are.

## Connection to psau-security

The scanner gets its data from the **psau-security** system (`native-app`, on Railway). Every lookup at the gate is local, so the network is only used for syncing: a full copy about once a day and the changes every 4 hours (or **Sync Now**).

- **Accounts**: guards sign in with their **existing psau-security account** (same email and password as the website and the mobile app) through psau-security's normal `POST /api/login`. There are no separate scanner accounts. Only staff roles (`security`, `admin`, `system_admin`) are accepted; a student/vehicle-owner account is refused and the session it opened is closed again. The guard's name is recorded with every acknowledged violation.
- **Data**: psau-security's read-only gate endpoints (`native-app/src/Controllers/Api/GateScannerApiController.php`), staff roles only:

  | Endpoint | Returns |
  |---|---|
  | `GET /api/security/gate/vehicles?updated_since=&page=&per_page=` | vehicles with owner name, contact, photo, colour/make/model, registration status. Deltas include removed vehicles (`removed: true`) |
  | `GET /api/security/gate/violations?updated_since=&page=&per_page=` | full sync: every **unsettled** violation. Deltas: every violation or sanction that changed, with `is_active`, so lifted suspensions, approved appeals and deletions clear on the laptop too |
  | `GET /api/security/gate/owner-photo/{userId}`, `.../violation-photo/{violationId}` | photos, downloaded once for offline use |

- **Who triggers the alert**: the same rule as psau-security's `SanctionService::hasUnsettledFor()`: a violation still waiting for its sanction, or an active Suspended/Revoked sanction. A suspension stops alerting once its end date has passed, on the server and on the laptop (even if it has not synced since).
- **Same plate twice**: psau-security treats `ABC-1234` and `ABC 1234` as different vehicles; the scanner reads them as the same plate. Violations of either still alert, and the dashboard shows the owner of the vehicle that has the violation.

## Packaging

```powershell
.\build.ps1
```

This fetches the plate models into `models\alpr\` if needed (the gate laptop is offline, so the models are bundled), runs the tests, and builds `dist\PlateScanner\PlateScanner.exe`. Copy the whole `dist\PlateScanner` folder to the target computer (a USB drive or shared folder both work — everything the app needs is inside it).

The models run on ONNX Runtime (no PyTorch), so the build is small and starts quickly.

### Using it on another computer

1. Copy the whole `dist\PlateScanner` folder to the other computer.
2. From that folder, run `..\make_shortcut.ps1 -ExePath PlateScanner.exe` (or point `-ExePath` at wherever you copied `PlateScanner.exe`) to add a **PSAU Gate Plate Scanner** shortcut to the Desktop.
3. Double-click the shortcut to launch the app. On first run it writes its own `config.json` to `%LOCALAPPDATA%\PlateScanner\` (see *Configuration*) — edit `camera.source` there for that computer's camera.

## Security notes

- The guard's API token is encrypted with Windows DPAPI (`session.bin`), so it's only readable by the same Windows user on that laptop.
- An expired token (HTTP 401) prompts for sign-in again. Scanning continues on the local database the whole time.
