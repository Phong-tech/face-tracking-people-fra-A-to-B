# MacBook People Counter (anonymous, with Arduino indicator)

A student smart-city IoT project. The MacBook camera detects and tracks people
with YOLO, counts them entering and leaving across a virtual line, logs the
anonymous counts to CSV and sends the current occupancy to an Arduino UNO WiFi
Rev2, which shows it with green / blue / red LEDs and a warning beep.

```
 Camera ──► YOLO (person only) + tracker ──► line-crossing logic ──► occupancy + peak
                                                                         │
                         ┌───────────────────────────────────────────────┤
                         ▼                       ▼                       ▼
                 live window (HUD)      CSV logs in data/      "COUNT:n" over USB serial
                                                                         ▼
                                                              Arduino LEDs + buzzer
```

## Privacy

* All processing is **local**. Nothing is sent to any server (the YOLO weights are
  downloaded once on first run; no camera data ever leaves the Mac).
* **No facial recognition.** The system only detects "a person" and never
  identifies or classifies individuals.
* **No video or photographs are saved.** Frames exist only in memory while they
  are shown in the window. The code contains no image- or video-writing calls.
* Only anonymous counts, timestamps and **temporary tracking IDs** are stored. A
  tracking ID is just a number the tracker hands out during one running session;
  it does not represent a person and is meaningless after the program closes.
* **Tell people** when a camera-based counter is being tested (for example with a
  sign at the entrance) and follow your school's / country's rules (GDPR in
  Norway/EU). Test with consenting classmates, not the public, unless approved.
* The project needs no API keys or Wi-Fi passwords. Do not add any to the repository.

## Project structure

```
macbook_people_counter/
├── people_counter.py        Main app: camera, YOLO tracking, window, keys
├── generate_report.py       Statistics + hourly CSV + PNG chart (no camera needed)
├── config.json              All settings
├── requirements.txt         Python dependencies
├── pytest.ini               Test settings
├── README.md                This file
├── .gitignore
├── data/                    CSV logs are created here (.gitkeep keeps the folder)
├── arduino/occupancy_indicator/occupancy_indicator.ino   Arduino sketch
├── src/
│   ├── config.py            Loads and validates config.json
│   ├── counter.py           Line-crossing + occupancy + peak logic (no camera code)
│   ├── serial_manager.py    Arduino port detection, reconnect, COUNT messages
│   ├── logger.py            CSV logging and restoring totals after a restart
│   └── overlay.py           Drawing the line, boxes and on-screen numbers
└── tests/
    ├── test_counting_logic.py       Counting, dead zone, expiry, serial messages
    └── test_logging_and_config.py   CSV restore, config validation, report
```

## Setup on macOS

Requires Python 3.11 or 3.12 (`brew install python@3.12` if needed).

```bash
cd macbook_people_counter
python3.12 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 people_counter.py
```

The first run downloads `yolo11n.pt` (about 6 MB) automatically, so you need
internet once. The first start-up can take a while because PyTorch is loaded.

### Camera permission

1. Run the program from Terminal (or your IDE). macOS shows "Terminal would like
   to access the camera" – click **OK**.
2. If you clicked "Don't allow": open **System Settings → Privacy & Security →
   Camera**, switch on the app you run Python from (Terminal, iTerm, VS Code,
   PyCharm), then restart that app.
3. Close other apps that use the camera (Zoom, FaceTime, Photo Booth).

### Keys in the camera window

| Key | Action |
|-----|--------|
| `Q` or `Esc` | Quit |
| `R` then `Y` | Reset counters. `R` asks for confirmation; press `Y` within 5 s. Any other key cancels. |
| `W` / `S` | Move a **horizontal** line up / down |
| `A` / `D` | Move a **vertical** line left / right |
| `F` | Flip which direction counts as ENTER (and EXIT) |

Moving the line or flipping with keys lasts until you quit. To keep it, copy the
`Line:` value shown in the window into `config.json` (see below).

## How the Arduino is wired

Use the **UNO WiFi Rev2** pins below. Every bare LED **must** have its own
current-limiting resistor (220–330 Ω is fine at 5 V) so neither the LED nor the
pin is damaged.

| Part | Arduino pin | Wiring |
|------|-------------|--------|
| Red LED   | **10** | pin 10 → 220 Ω resistor → LED long leg (anode); short leg (cathode) → GND |
| Blue LED  | **9**  | pin 9 → 220 Ω resistor → LED long leg; short leg → GND |
| Green LED | **8**  | pin 8 → 220 Ω resistor → LED long leg; short leg → GND |
| Buzzer    | **3**  | pin 3 → buzzer (+); buzzer (–) → GND (a 100 Ω series resistor is a good idea) |

All grounds go to a GND pin on the Arduino. The Arduino is powered and
communicates through the USB cable to the Mac – no extra power is needed.
The sketch assumes a *passive* buzzer; for an *active* buzzer set
`BUZZER_IS_PASSIVE = false` in the sketch.

LED meaning: **green** below the low threshold, **blue** between low and medium,
**red** at or above medium. At maximum capacity the red LED stays on and a short
beep sounds (once per newly arriving person, never continuously). If no valid
message arrives for 6 s, all three LEDs blink (lost communication). They also
blink at power-up until the first message arrives.

### Upload the sketch

1. Install the **Arduino IDE** and, in *Tools → Board → Boards Manager*, install
   **Arduino megaAVR Boards** (this is the package for the UNO WiFi Rev2).
2. Connect the board by USB. Choose *Tools → Board → Arduino megaAVR Boards →
   Arduino Uno WiFi Rev2* and the port (looks like `/dev/cu.usbmodem…`).
3. Open `arduino/occupancy_indicator/occupancy_indicator.ino` and click **Upload**.
4. Quick hardware test: open the Serial Monitor (9600 baud, line ending
   **Newline**), type `COUNT:3`, `COUNT:7`, `COUNT:12`, `COUNT:15` and watch the
   LEDs change. **Close the Serial Monitor before running `people_counter.py`**,
   because only one program can use the port at a time.

The thresholds at the top of the sketch (`LOW_THRESHOLD`, `MEDIUM_THRESHOLD`,
`MAX_CAPACITY`, `COMM_TIMEOUT_MS`) must match `config.json`.

### Find the Arduino serial port

By default the program searches for `/dev/cu.usbmodem*` and `/dev/cu.usbserial*`
automatically. To see the name yourself:

```bash
ls /dev/cu.usb*
```

To force a specific port, set it in `config.json`, for example
`"serial_port": "/dev/cu.usbmodem14101"`.

## Run without the Arduino

Nothing special is needed: if no Arduino is found the program keeps counting and
logging and retries every few seconds (the window shows "not connected").
To turn the Arduino off completely:

```bash
python3 people_counter.py --no-arduino
```

(or set `"serial_enabled": false` in `config.json`).

## Configuration (`config.json`)

| Setting | Meaning |
|---------|---------|
| `camera_index` | Which camera. `0` = MacBook camera. Also `--camera 1` on the command line. |
| `detection_confidence` | 0–1. Higher = fewer false detections but more missed people. Start at 0.4. |
| `line_orientation` | `"horizontal"` (people cross up/down) or `"vertical"` (left/right). |
| `line_position` | Where the line is, as a fraction of the frame (0.05–0.95). 0.5 = middle. |
| `entry_direction` | Which movement is ENTER. Horizontal: `"down"` or `"up"`. Vertical: `"right"` or `"left"`. |
| `dead_zone_fraction` | Half-width of the dead band around the line (fraction of frame size). 0.05 on a 720 px high frame ≈ 36 px each side. |
| `reference_point_ratio` | Which part of the box is tracked: 0.85 = near the feet. |
| `tracking_timeout_seconds` | Forget a tracking ID not seen for this long. |
| `serial_port` | `"auto"` or a path such as `/dev/cu.usbmodem14101`. |
| `baud_rate` | Must match the sketch (`9600`). |
| `serial_settle_seconds` | Wait after opening the port, since the Arduino resets. |
| `heartbeat_interval_seconds` | `COUNT:n` is re-sent this often even if unchanged. |
| `low_occupancy_threshold` / `medium_occupancy_threshold` / `max_capacity` | Level boundaries (see LEDs). Need `low < medium ≤ max`. |
| `log_interval_seconds` | How often a snapshot goes to `occupancy_log.csv`. |
| `device` | `"auto"`, `"cpu"` or `"mps"` (Apple Silicon GPU, often faster). |

Invalid values stop the program with a clear message instead of misbehaving later.

### Change the counting line
Edit `line_orientation` and `line_position` in `config.json`, or nudge it live
with `W`/`S` (horizontal) or `A`/`D` (vertical) and copy the shown value to the config.

### Reverse entry and exit
Swap `entry_direction` (`"down"` ↔ `"up"`, or `"right"` ↔ `"left"`), or press `F`
in the window. The green ENTER arrow always shows what is currently counted as entering.

### Adjust detection confidence
Change `detection_confidence`. Ghost boxes → raise it (0.5). People missed → lower it (0.3).

### Change capacity thresholds
Edit the three threshold values in `config.json` **and** the same constants at
the top of the Arduino sketch, then re-upload the sketch.

## How counting works

1. YOLO finds people (class 0 only) in every frame and the tracker keeps a
   temporary ID for each person.
2. For each person we take **one reference point**: the horizontal centre of the
   box, 85 % of the way down it (near the feet, i.e. their place on the floor).
3. The distance of that point to the line decides the person's side:
   clearly *before* the line, clearly *after* it, or *inside the dead zone*.
4. A crossing is counted only when a person's last **clear** side differs from
   their new **clear** side. While in the dead zone nothing changes, so standing
   on the line or wobbling around it cannot add counts (hysteresis).
5. After a crossing the new side is remembered, so the same crossing is never
   counted again. Walking back across the line is a genuine second crossing and
   is counted as the opposite direction.
6. Moving in the `entry_direction` is **ENTER**, the opposite is **EXIT**.
7. IDs not seen for `tracking_timeout_seconds` are removed from memory. If the
   same ID reappears it is treated as a new track and only counts after it
   has been clearly seen on one side and then the other.
8. `current occupancy = total entered − total exited`. An EXIT while the
   occupancy is already 0 is ignored (and reported in the terminal), so it can
   never go negative.

### How the peak is calculated
Every time an ENTER makes the occupancy larger than the highest value so far,
the peak and the **local date-time of that moment** are updated. The peak
timestamp is therefore the *first* time the highest value was reached.
At start-up the totals and peak are rebuilt by replaying `crossing_events.csv`.

## Data files

* `data/crossing_events.csv` – `timestamp,tracking_id,direction,current_occupancy`;
  one row per confirmed crossing, written immediately. A manual reset adds a
  `RESET` row (empty tracking ID) so later restarts do not resurrect old totals.
* `data/occupancy_log.csv` – `timestamp,current_occupancy,total_entered,total_exited,peak_occupancy,peak_timestamp`;
  one row every `log_interval_seconds` and one when the program stops.

Files are appended to and flushed on every write; previous data is never overwritten.
Timestamps use your Mac's local time. The CSVs are in `.gitignore` by default.

## Generate the statistics report

```bash
source .venv/bin/activate
python3 generate_report.py
```

It prints total entries, total exits, final occupancy, highest occupancy and its
exact date-time, the busiest hour and the average occupancy by hour. It also writes
`data/hourly_summary.csv` and `data/occupancy_chart.png`. It does not open the
camera, so it can run any time (use `--no-chart` to skip the PNG, `--data-dir`
to read another folder).

"Busiest hour" is the hour of the day with the highest *average occupancy* in
the periodic observations (if there are none, the hour with the most entries).

## Run the tests

```bash
source .venv/bin/activate
python3 -m pytest -v
```

The tests need no camera, Arduino or YOLO model. They cover crossing in both
directions, staying on one side, the dead zone, repeat-count prevention,
non-negative occupancy, ID expiry, rejection of invalid serial messages, the
serial reconnect logic, CSV restore, configuration validation and the report.

## Troubleshooting

**The camera will not open**
* Grant Camera permission (see above) and restart Terminal/IDE.
* Quit other apps using the camera.
* Try `python3 people_counter.py --camera 1` (an iPhone via Continuity Camera or
  an external webcam can take index 0 or 1).
* Make sure no second copy of the program is running.

**Counts are inaccurate or duplicated**
* Place the line where people walk through in a clear direction, not where they
  stop or turn. Make sure their feet/lower body are visible – the reference
  point is near the bottom of the box.
* Increase `dead_zone_fraction` (e.g. 0.08) if people loiter near the line.
* Adjust `detection_confidence`; improve lighting; avoid strong backlight.
* People walking close together or in front of each other can merge into one
  box or swap IDs and be mis-counted – this is a limitation of tracking.
* Increase `tracking_timeout_seconds` if people are briefly lost behind others.
* Watch the terminal: every counted crossing is printed with its tracking ID.
  Compare it with a manual count for several test runs to measure accuracy
  yourself – no accuracy figure is claimed here.
* Low FPS makes fast walkers jump over the dead zone; try `"device": "mps"`
  or a smaller `inference_image_size` (e.g. 480).

**The YOLO model is unavailable**
* First run needs internet to download `yolo11n.pt`. Check your connection,
  VPN or proxy.
* Download `yolo11n.pt` manually, place it in the project folder and keep
  `"model_path": "yolo11n.pt"`, or set `model_path` to its full path.
* `ModuleNotFoundError: ultralytics` → activate the virtual environment and run
  `python3 -m pip install -r requirements.txt`. Use Python 3.11/3.12.

**The Arduino does not connect**
* Check the cable (some are charge-only) and run `ls /dev/cu.usb*`.
* Close the Arduino IDE Serial Monitor or any other program using the port
  (the error "Resource busy" means this).
* Set `serial_port` explicitly in `config.json`.
* Confirm the sketch is uploaded and `baud_rate` is 9600 in both places.
* Wait a few seconds after connecting: the board resets when the port opens.
* Remember the program keeps counting without the Arduino and reconnects by itself.

## Limitations

* Accuracy has **not** been measured for this project. Test it in your own
  setting and report your own results.
* A single MacBook camera has a narrow field of view and a fixed angle;
  crowds, occlusion, backpacks, strollers, unusual poses and poor light reduce reliability.
* People who stay in view but are hidden for longer than the tracking timeout,
  or whose ID changes while they stand in the dead zone, may be counted wrongly
  or missed.
* The count is only as good as the starting point: if people are already inside
  when you start, occupancy begins at 0 (or the restored value), and exits of
  people who were never counted as entering are ignored.
* Peak time is based on counted entries, not ground truth.
* Everything runs on the Mac; if the Mac sleeps or the program stops, counting stops.
* The Arduino thresholds are constants in the sketch and must be kept in sync
  with `config.json` by hand.
