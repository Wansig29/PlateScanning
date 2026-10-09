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
.\.venv\Scripts\python tools\seed_demo.py                       # demo vehicles + violations + demo guard
.\.venv\Scripts\python tools\make_test_video.py gate.mp4 NBC1234 ABC1234 XYZ789 QWE4567 --no-stop --blur --gap 0.6 --lanes 2
.\.venv\Scripts\python -m platescanner --source gate.mp4        # enter demo@psau.local / demo, then "Continue offline"
```

`make_test_video.py` options: `--no-stop` (drive through instead of stopping), `--speed` (px/frame; 18 ≈ 6.5 km/h, 45 ≈ 16 km/h), `--gap` (seconds between cars, so several are in view), `--lanes 2`, `--blur` / `--shutter` (motion blur from speed × exposure time).

The demo plates show every result type:
- `NBC1234` has one active violation.
- `ABC1234` has two active violations, so the pager appears.
- `XYZ789` is registered with no active violation.
- `QWE4567` is not registered.

## Using a normal 1080p webcam

The scanner is built to work with an ordinary USB webcam, so the camera's own settings decide a lot:

1. **Run the probe first**, with the camera mounted where it will stay, pointed at the lane, in the usual light: `python tools\camera_probe.py`. It measures the frame rate the camera really delivers in each video mode, sweeps the shutter speed (exposure), saves a sample picture of each and prints a `camera` block for `config.json`. Many webcams manage only a few frames per second at 1080p unless set to **MJPG**, which is now the default (`camera.fourcc`).
2. **Shorter shutter = sharper plates.** Auto-exposure picks a slow shutter, the main cause of motion-blurred plates. Set `camera.exposure` (on Windows: -6 = 1/64 s, -7 = 1/128, -8 = 1/256, -9 = 1/512) to the shortest value at which the picture is still bright enough; `camera.gain` can brighten it at the cost of noise. The app logs what the driver actually accepted when the camera opens, and warns when it ignores a setting.
3. **Fix the focus.** Autofocus can hunt as vehicles pass: set `camera.autofocus` to false and `camera.focus` once the camera is mounted.
4. **Limits to expect.** A webcam shutter and no infrared mean reading is reliable in daylight at walking-to-slow-vehicle speeds, and degrades with speed, at night and in glare (see the speed figures under *Motion blur*). A lamp aimed at where plates pass, a slowdown at the gate and a camera close to the lane help more than any software setting. `health.py` warns when the picture gets too dark, blurred or slow.
5. **Existing installs keep their saved `config.json`**, which says 1280x720 and no video format. Delete the `camera` block (or set `width` 1920, `height` 1080, `fourcc` "MJPG") to pick up the new defaults.

## Configuration

On first run, the app writes `config.json` to `%LOCALAPPDATA%\PlateScanner\`. You can point it somewhere else with the `PLATESCANNER_HOME` environment variable. The same folder holds the database, the downloaded photos, the plate captures and the logs.

| Setting | Meaning |
|---|---|
| `api.base_url` | psau-security address (default `https://psau-security-production.up.railway.app`) |
| `api.login_path`, `vehicles_path`, `violations_path` | Endpoint paths (see *Connection to psau-security*) |
| `camera.source` | `"0"` for the first USB camera, an `rtsp://…` URL for an IP camera, or a video file |
| `camera.roi` | `[x, y, w, h]` fractions of the frame to watch, e.g. `[0.2, 0.4, 0.6, 0.6]`, so trees or passers-by outside the lane don't trigger scans |
| `motion.min_area_ratio` | How much of the ROI must move to count as a vehicle |
| `motion.travel_frames`, `min_travel_ratio`, `min_growth_ratio` | The moving shape must also travel: over the last `travel_frames` frames its centre moves at least `min_travel_ratio` of the picture width, or its area grows by `min_growth_ratio` (driving up to the camera). Someone fidgeting in place doesn't count. Raise `min_travel_ratio` if people still trigger scans; lower it if slow vehicles are missed |
| `ocr.plate_layouts` | Accepted plate formats (`L` = letter, `D` = digit) |
| `ocr.detector_model` | Plate detector input size. `…-t-384-…` (default) is the best speed/recall trade-off on a CPU; `…-t-512-…` / `…-t-640-…` find smaller, farther plates but are slower |
| `ocr.ocr_model` | Plate OCR models, comma-separated. Each plate is read by all of them and the results are merged character by character |
| `ocr.read_confidence`, `report_confidence`, `alert_confidence`, `confirm_reads` | When a vehicle is reported (see *Deciding a plate*) |
| `ocr.detector_confidence` | How sure the neural detector must be that something is a plate (default 0.5). Raise it if shelves, windows or signs are still read as plates; lower it if real plates are missed |
| `ocr.classical_only_confidence` | A candidate that only the classical finder proposed must read at least this well (default 0.75) before it is reported |
| `ocr.classical_proposals` | Also try plate candidates from the classical finder (catches plates the neural detector misses, ~20 ms per frame) |
| `scan.archive_after_days` | Pictures older than this many days are **moved** (never deleted) from `captures\` to the archive folder; the Logs still open them. Default 30; 0 keeps everything in `captures\` |
| `scan.archive_dir` | Where the old pictures go. Empty = `archive\` next to `captures\`; can be another drive, e.g. `D:\\PlateScannerArchive` |
| `scan.archive_ended_academic_year` | `true` (default): once an academic year has ended, its scan log is archived automatically: a CSV in `archive\<year>\`, and its scans leave the Logs and appear under *Reports → Archive*. Nothing is deleted |
| `scan.academic_year_start_month` | Fallback only, used until the school years have synced from psau-security: the month the academic year starts (default 8 = August; 1 = a calendar year) |
| `scan.save_pictures_for` | Which results get pictures saved (the log row is always kept). Default `["violation"]`: pictures are what take the disk space, so only the evidence is kept. Add `"no_plate"`, `"clear"` or `"not_registered"` to keep those too |
| `scan.plate_cooldown_seconds` | Ignore repeat reads of the same plate while it's still at the gate |
| `scan.fuzzy_match` | Accept a read that's one character off, if exactly one plate matches. The UI flags these as *approximate* |
| `scan.require_acknowledge` | `false` (default): a violation alert flashes and sounds once, then the next scan replaces it; every scan is still logged. `true`: the alert stays until a guard acknowledges it (the options below then apply) |
| `scan.reminder_seconds` | Repeat the alarm this often while a violation is unacknowledged (0 = alert once only) |
| `scan.bring_to_front` | Bring the app to the front on every violation alert |
| `sync.interval_hours` / `full_resync_hours` | Sync every 3 h: vehicles as a delta, violations always as the complete list of unsettled ones, so a violation deleted online stops alerting at the next sync (nothing is written when nothing changed). A full re-download of the vehicles every 24 h drops vehicles deleted online without a `removed` record |

Keyboard: **F11** toggles full screen. `--fullscreen` starts the app in full screen.

## How it works

- **Newest frame only** (`pipeline.py`): the camera thread hands the recognizer only the latest frame; older ones are dropped. Recognition therefore always works on what is at the gate *now* and never builds up a backlog in heavy traffic. Test videos are played the same way (in real time, dropping frames when behind), so they behave like a live camera.
- **Motion gate** (`vision/motion.py`): MOG2 on a 320 px copy of each frame. Only vehicle-like motion counts: one solid moving shape covering at least `motion.min_area_ratio` of the picture. People (shapes taller than `motion.max_height_ratio` × their width, or anything that moves about in place without travelling across or toward the camera), leaves and rain (specks that are mostly empty space when grouped) and lighting changes (more than `motion.max_area_ratio` of the picture at once) are ignored. When there is no vehicle motion and no real plate is being followed, plate detection runs only about once a second, so an empty gate costs almost no CPU. That once-a-second look also catches a vehicle the motion rules missed. A plate that was seen but never readable is logged as "plate not readable" only if vehicle-like motion happened while it was tracked, so a static or false detection no longer fills the Logs.
- **Assembly line**: the next frame goes through the plate detector while the plates of the current frame are being read, and all plates in a frame are read in parallel. On an 8-core laptop this raised the scanned frames from about 14 to 16–18 per second.
- **Plate detection** (`vision/alpr.py`): a YOLOv9 plate detector (open-image-models, ONNX, MIT license) finds every plate in the frame. The classical contrast/edge finder (`vision/plate.py`) adds candidates the network misses. Wrong candidates are harmless: their text never fits a plate layout.
- **One track per vehicle** (`vision/tracker.py`): detections are matched frame to frame by overlap and predicted motion, so each vehicle keeps its ID (`#12`) while it crosses the picture. Several vehicles, side by side or bumper to bumper, are followed at once. Plates cut off by the edge of the picture are tracked but not read until they are fully in view.
- **Motion blur** (`ocr.deblur`): a plate whose sideways edges are much weaker than its up-and-down ones is blurred by the vehicle's motion. Its blur length is measured (cepstrum) and undone (Wiener deconvolution) before reading, because blurred plates are often read *confidently wrong*. On test videos at 16–25 km/h this took reads from 4/8 and 0/8 to 8/8 and 7/8. At 40–50 km/h the camera's shutter decides: 8/8 at 1/1000 s, 7/8 at 1/250 s, 5/8 at a typical webcam's 1/100 s.
- **Plate OCR** (fast-plate-ocr, ONNX, MIT license): each plate crop is read by two small OCR models that tend to make different mistakes. Their outputs are layout-corrected (`A8C1234` → `ABC1234`) and merged character by character. On the test set this pair reads more plates correctly than the single model ten times its size.
- **Deciding a plate**: all reads of one vehicle are combined character by character, weighted by confidence, so a frame that misreads one letter is outvoted by the others. A vehicle is reported as soon as the evidence is strong enough:
  - a **violation** on the first read that is at least `alert_confidence` (0.75) sure, or on `confirm_reads` (2) agreeing reads;
  - anything else once `confirm_reads` reads agree;
  - if the vehicle leaves first, its best guess is reported if it averages `report_confidence` (0.65 by default; a violation only needs `read_confidence`, because missing a violator is worse than a doubtful alert). Otherwise it is logged as *plate not readable*, with a snapshot.

  After reporting, a vehicle is still re-read now and then. If the consensus clearly changes, a correction is logged.
- **Database-aware decoding** (`decode.py`): the OCR models output a probability for every character at every position, not just the winner. These are kept, combined over the vehicle's reads (repeat frames count for less, since their errors are correlated) and scored against every registered plate, weighing "one of our vehicles" against "a visitor". A registered plate replaces the plain read only if **all** hold: it wins with at least `ocr.decode_accept` (0.90) certainty; it differs from the plain read in at most `decode_max_changes` (2) characters; every character it changes was at least `decode_min_char_prob` (10%) likely to the OCR itself, so a confident read is never overridden by a nearby registered plate; and its registered colour doesn't clearly contradict the colour seen at the gate (a mild penalty, `decode_colour_penalty`). The result is flagged *approximate*. This resolves two reads that disagree on one doubtful character without waiting for the vehicle to leave. `decode_temperature` (2.0) softens the OCR's overconfidence and should be calibrated on real gate footage. Set `decode_with_database` to false to turn it off.
- **Lookup** (`db.py`): the plate exactly as read is looked up first. Only if it is not in the database is a confusion-folded key tried (O/D/Q/0, I/L/1, B/8, S/5…), then, with `fuzzy_match`, one character off. Look-alike plates such as `ABD 1234` and `ABO 1234` are different real plates, so such a match is flagged *approximate* (and a violation **VERIFY PLATE**), and a vehicle only ever shows the violations recorded against it or its exact plate, never those of a look-alike. Only local SQLite is queried, never the API.
- **Suspension dates**: synced dates are stored as ISO (`2026-10-10`); `10/10/2026` or `Oct 10, 2026` are converted. A date that still can't be read keeps the violation alerting until the next sync instead of silently ending it.
- **Errors never stop the alert**: if a picture can't be saved (disk full) the scan is logged and alerted without it; if the scan log can't be written (database locked) the guard is still alerted. If plate reading stops anyway, the scanner beeps and says so instead of leaving only the live feed running.
- **Live feed**: every tracked vehicle gets a box, a trail and a label (`#12 ABC 1234 98%`, or `#13 reading...` until decided). Between recognitions, boxes move with the vehicle's measured speed so they stay on the plate. While a **violator** is in view the rest of the picture is dimmed and the violator's whole vehicle gets a thick, blinking red frame, so it stands out from the cars around it.
- **Several violators at once**: when acknowledgement is not required (the default), each violator in view gets its own card on the Identity Dashboard (up to 3, newest first), with its photo, owner, violation and suspension dates. A "2 violators in view  ‹ ›" bar above the cards jumps between them when they do not all fit. A card stays while its vehicle is in view and for 10 s after; a clear or unregistered vehicle never pushes a violator off the screen. Everything is still logged.
- **Which vehicle is it?** (`vision/identify.py`): with several cars at the gate a plate number alone is hard to match to a moving car, so the dashboard leads with:
  - **This vehicle**: a photo of the vehicle itself, cut from the camera frame;
  - **Where it was**: the whole scene at that moment with only this vehicle highlighted (others outlined in grey);
  - its **colour**, its **position** in the picture (left / middle / right), how many **other vehicles** were in view, and its live status: **In view now**, or *Left the picture 8 s ago*;
  - the banner names the vehicle number (`VEHICLE #12`) that labels it on the live feed.

  Owner, violation and suspension follow in one compact card; the violation's evidence photos open from a link. The Logs and Captured Plates show the colour and position too (*Red, left side · Illegal parking*), and all of it is saved with the scan, so reopening a scan from the Logs shows the same pictures.
- **No violator goes unnoticed** (only when `scan.require_acknowledge` is `true`; off by default, in which case the alert clears itself and the scan is just logged): every violation alert stays open until a guard acknowledges it.
  - The alert itself: the dashboard banner turns red and flashes, an alarm sounds, the taskbar flashes, the dashboard expands if collapsed, and the app **brings itself to the front** if another window covers it (`scan.bring_to_front`; Windows may only flash the taskbar when another program has the focus).
  - While anything is unacknowledged: a **pulsing red frame** runs around the whole window (visible from across the booth), a red **"⚠ N unacknowledged violations"** counter sits in the top bar, and the whole alert **repeats every `scan.reminder_seconds`** (15 s) until someone responds.
  - The violator **stays on the dashboard** until acknowledged; however much traffic follows, other vehicles never replace it.
  - **Several violators at once**: each gets its own alarm (played one after another, never on top of each other). The dashboard shows the first and queues the rest (*Acknowledge violation · 2 more waiting*); each Acknowledge brings up the next. Opening an older scan from the Logs keeps the queue and offers *Back to violations*.
  - **Who acknowledged what, and when** is stored with the scan (the signed-in guard's name) and shown in the Logs (*✓ acknowledged by Juan at 10:05:12*, or a red *⚠ NOT ACKNOWLEDGED*).
  - It **survives a restart**: violations nobody acknowledged in the last `scan.unacknowledged_lookback_hours` (24 h) are raised again when the app starts (unless the violation was resolved online in the meantime).
- **Snapshots, not video**: no video is recorded. Each reported violator saves three JPEGs (the plate crop, the vehicle photo and the scene with the vehicle highlighted, `…_scene.jpg`) to `captures\<result>\YYYY-MM-DD\`, where `<result>` is `violation`, `no_violation`, `not_registered` (the plate is not in the database) or `no_plate_read`.
- **Reports** (status bar → *Reports*): the scans of the last day, week, month or year (the same periods as psau-security's violation map), with a count per result, a filter, *Open pictures folder* and *Export to CSV*. The **Archive** button lists the academic years that were archived automatically after they ended. Plates that were seen but never readable are logged too; click the row to see the snapshot.
- **Chat-style feeds**: Logs and Captured Plates add new entries at the bottom and auto-scroll. Scrolling up pauses this and shows a "▼ N new scans" button.

- **"Verify plate" for look-alike registered matches**: a vehicle that matched a registered plate only as a look-alike or one character off (see *Lookup*) is never shown as a plain green *no violation*: its banner is amber and says **VERIFY PLATE**, and its Logs row too, since it may be an unregistered car with a similar plate.
- **"Verify plate" alerts**: a violation that rests on a doubtful read (an approximate or decoded match, an average confidence under `ocr.verify_below_confidence` (0.60), or a single read under `ocr.verify_single_read_below` (0.90)) is still raised at once, but its banner and Logs row say **VERIFY PLATE**, so the guard compares the plate with the photo of the vehicle instead of trusting it blindly.
- **Health warnings** (`health.py`): the status bar warns when the camera image is blurred or dirty, the frame or analysis rate drops, the picture freezes, the scene is too dark or overexposed, or more than half of the recent plates could not be read. Each warning shows once, a stalled feed also beeps, and a green *Recovered* follows when it clears.

### Measuring speed and accuracy

```powershell
.\.venv\Scripts\python tools\make_test_video.py rush.mp4 NBC1234 ABC1234 XYZ789 QWE4567 DEF5678 GHJ2345 --no-stop --blur --speed 30 --gap 0.8 --lanes 2
.\.venv\Scripts\python tools\bench_live.py rush.mp4
.\.venv\Scripts\python tools\bench_live.py real_gate.mp4 --set camera.roi=[0.1,0.3,0.8,0.7]   # real footage: lists every read
```

To check that `ocr.plate_layouts` covers the plates you actually have, run `python tools\plate_formats.py`: it lists the plate shapes in the local database (L = letter, D = digit), which ones the layouts miss, and a suggested list. It only reads the database.

For accuracy on **real footage**, write a CSV of what each video really shows (`video, seconds_start, seconds_end, plate, condition`, with the condition free text such as `night` or `rain`) and run:

```powershell
.\.venv\Scripts\python tools\eval_footage.py footage_folder truth.csv --out eval_out
```

`eval_out
eport.md` gives, per condition and overall, the vehicles read exactly right, read wrong (expected vs got), missed and falsely reported, plus the character error rate; `mismatches.csv` lists every miss. It runs on a temporary database and never touches the real scan log. `--selftest` shows the report format without any footage.

`bench_live.py` plays a video in real time through the same threads the app uses and reports, for each vehicle, whether it was read, whether correctly, and how long after its plate appeared. Run it on footage from the actual gate camera before deploying, and use `--set section.name=value` to try settings.

### Freeing disk space: delete pictures of non-violations

Pictures of clear, unregistered and unreadable scans are no longer saved (`scan.save_pictures_for`). To delete the ones saved earlier, with the scanner closed:

```powershell
.\.venv\Scripts\python tools\purge_pictures.py          # shows what would be deleted, deletes nothing
.\.venv\Scripts\python tools\purge_pictures.py --yes    # deletes it
```

Only pictures go. The log rows stay, so the Logs, Reports and CSV exports are unchanged, and violation pictures are never touched. It searches `captures\` and the archive folder.

To clear test scans entirely, the log rows and all pictures (violations too) dated before a day:

```powershell
.\.venv\Scripts\python tools\purge_pictures.py --delete-before 2026-10-01          # shows what would go
.\.venv\Scripts\python tools\purge_pictures.py --delete-before 2026-10-01 --yes    # deletes it
```

To clear just one day (for example today's test scans), with its pictures:

```powershell
.\.venv\Scripts\python tools\purge_pictures.py --delete-today          # shows what would go
.\.venv\Scripts\python tools\purge_pictures.py --delete-today --yes    # deletes today's scans
.\.venv\Scripts\python tools\purge_pictures.py --delete-on 2026-10-06 --yes   # a specific day
```

This cannot be undone. Vehicles, violations and the sync data are not touched, only the scanner's own scan log. Archived CSV files are not edited.

### Resolution: does 1080p read better than 720p or 480p?

`tools/bench_resolution.py` renders one synthetic traffic scene at 1080p and downscales it (same field of view) to 720p, 480p and 360p, scans each through the real pipeline, and measures the single-crop reading rate against plate width in pixels. On that scene (10 cars, plates 59-270 px wide): exact reads 1080p 60%, 720p 60%, 480p 50%, 360p 30%, and **720p with a narrower field of view 80%**. Single crops first read 80-90% of the time at about 80-96 px plate width, and voting over several frames read some plates down to about 46 px. So what matters is the **pixels across the plate**, not the camera's resolution on its own: a tighter shot of the lane beats a wider shot at higher resolution. Keep the camera close, point it at the lane and set `camera.roi` to the lane (the detector sees only that area, so plates are bigger to it). The status bar warns when the median plate is narrower than `ocr.min_plate_width_px` (80). Caveats: the footage is synthetic and its rendered cars suit the plate detector poorly (most reads came from the classical finder), so re-measure on real footage before quoting these figures.

### Tried and not used: deskew and contrast enhancement

`ocr.deskew` (straighten tilted plates) and `ocr.enhance` (fix dark, low-contrast or blown-out crops), in `vision/enhance.py`, are off by default. `tools/bench_conditions.py` (synthetic plates, 300 per condition, the real OCR models) found no worthwhile gain: deskew lowered reads on rotated plates by about 1.5 points on average, because the OCR already tolerates tilt and resampling blurs, and enhance changed them by +0.2 points (noise). They are kept so the comparison can be re-run on real footage.

### Tried and not used: multi-frame pixel fusion

`vision/fusion.py` aligns several crops of one plate and takes their per-pixel median, and `tools/bench_fusion.py` compares it with the voting the app already does (synthetic plates, 3-8 frames each, the real OCR models). Combining the *pixels* never beat voting over the *reads* (exact reads at 56 / 80 / 120 px plate width: single frame 24 / 58 / 86%, vote 35 / 73 / 96%, fused 24 / 64 / 91%), so it is not wired in. The same run shows how steeply reading depends on plate width: nothing reads below about 40 px, and reliable reading needs plates well over 80 px wide. Re-measure on real footage before relying on these figures.

### Multiple violations per vehicle

The spec left this open. The app stores **all** active violations. The dashboard shows the newest one in the wireframe's fields. When there are more, it rotates through them every 4 s, and the ‹ › buttons page manually (pausing rotation for 15 s). The banner and log entry say how many there are.

## Connection to psau-security

The scanner gets its data from the **psau-security** system (`native-app`, on Railway). Every lookup at the gate is local, so the network is only used for syncing: a full copy about once a day and the changes every 4 hours (or **Sync Now**).

- **Accounts**: guards sign in with their **existing psau-security account** (same email and password as the website and the mobile app) through psau-security's normal `POST /api/login`. There are no separate scanner accounts. Only staff roles (`security`, `admin`, `system_admin`) are accepted; a student/vehicle-owner account is refused and the session it opened is closed again. So is an account whose login response carries no role at all (`user.role`, or a `user.roles` list): the scanner fails closed rather than trusting it. The guard's name is recorded with every acknowledged violation.
- **Server text is plain text**: owner names, contact numbers, violation types and descriptions, and server error messages are shown exactly as typed, never interpreted as HTML (Qt labels guess HTML by default).
- **Offline sign-in**: when the server can't be reached, "Continue offline" still needs an email and password: those of a guard who signed in online on this laptop within `api.offline_login_days` (14). Only a salted PBKDF2 hash of the password is kept (`offline_guards.bin`, DPAPI-protected on Windows), never the password. The Database and Reports windows, which hold owners' names, contact numbers and photos, open only for a signed-in guard (online or offline); after *Sign out* they are locked again while scanning continues.
- **Data**: psau-security's read-only gate endpoints (`native-app/src/Controllers/Api/GateScannerApiController.php`), staff roles only:

  | Endpoint | Returns |
  |---|---|
  | `GET /api/security/gate/vehicles?updated_since=&page=&per_page=` | vehicles with owner name, contact, photo, colour/make/model, registration status. Deltas include removed vehicles (`removed: true`) |
  | `GET /api/security/gate/violations?page=&per_page=` | every **unsettled** violation, fetched in full on every sync (never as a delta), so lifted suspensions, approved appeals and deletions clear on the laptop at the next sync. Records with `removed: true` or a `deleted_at` are ignored |
  | `GET /api/security/gate/school-years` | every school year with its start and end date (the ones the admin manages in Utilities), fetched on every sync. A year is archived once its end date has passed. If the server doesn't have this endpoint yet the scanner keeps working with `scan.academic_year_start_month` |
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

### Signing a release (in-app updates)

Installed scanners only install an update that is signed with the release key, so a stolen GitHub login can't push code to the gate laptops. The private key stays on your own computer, never on GitHub.

One time: `python tools\sign_release.py keygen` creates the key (passphrase-protected, in `%USERPROFILE%\.platescanner\`) and prints `RELEASE_PUBLIC_KEY = "..."`. Paste that line into `platescanner\updates.py`, commit, and back up the `.pem` file and passphrase. Until a key is set, the update banner only opens the release page.

Each release, after the workflow has published it:

```powershell
python tools\sign_release.py sign PlateScanner-Setup.exe --tag v1.2.0   # the installer downloaded from the release
```

then upload the `PlateScanner-Setup.exe.sig` it writes to the same release. The signature covers the installer's SHA-256 and its tag, so an older signed installer can't be passed off as a newer release.

## Security notes

- The guard's API token is encrypted with Windows DPAPI (`session.bin`), so it's only readable by the same Windows user on that laptop.
- An expired token (HTTP 401) prompts for sign-in again. Scanning continues on the local database the whole time.
