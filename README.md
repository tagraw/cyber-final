# HandFly 🤙✈️
**Gesture-controlled Crazyflie via AIDeck + MediaPipe**

---

## Architecture

```
AIDeckStream  ──►  GestureDetector  ──►  FlightController  ──►  Crazyflie
(TCP frames)       (MediaPipe +          (HighLevelCommander    (CRTP / 
                    temporal buffer)      + state machine)       Crazyradio PA)
                        │
                     Overlay
                   (cv2 window)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. AIDeck firmware
Flash the standard **WiFi streamer** firmware to your AIDeck.  
Default stream endpoint: `192.168.4.1:5000`  
Edit `config.py` → `AIDECK_IP` / `AIDECK_PORT` if yours differs.

### 3. Crazyradio PA
Plug in your Crazyradio PA. Update `Config.URI` to match your drone's address.

---

## Running

### With real hardware
```bash
python main.py
```

### Mock mode (webcam only, no drone required — great for testing gestures)
```bash
python run_mock.py
```

---

## Gesture Reference

| Gesture | Hand pose | Drone action |
|---------|-----------|--------------|
| **Point Up** | Index finger pointing upward | Move up |
| **Point Down** | Index finger pointing downward | Move down |
| **Point Left** | Index finger pointing left | Move left |
| **Point Right** | Index finger pointing right | Move right |
| **Point Forward** | Index finger toward camera | Move forward |
| **Point Back** | Index finger away from camera | Move backward |
| **Closed Fist** | All fingers curled | Hold position |
| **Open Palm** | All 5 fingers extended flat | Takeoff / Land toggle |
| **Thumbs Up** | Thumb up, fist closed | Ascend |
| **Thumbs Down** | Thumb down, fist closed | Descend |
| **Peace / V** | Index + middle extended | Toggle SLOW ↔ FAST speed |
| **Pinch** | Thumb tip touching index tip | Run square patrol route |
| **Circular CW** | Move wrist in clockwise circle | 360° yaw spin clockwise |
| **Circular CCW** | Move wrist counter-clockwise | 360° yaw spin counter-clockwise |

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `URI` | `radio://0/80/2M/E7E7E7E701` | Crazyflie radio URI |
| `DEFAULT_HEIGHT` | `0.5 m` | Takeoff altitude |
| `STEP_XY` | `0.3 m` | Horizontal step per command |
| `STEP_Z` | `0.2 m` | Vertical step per command |
| `GESTURE_HOLD_FRAMES` | `8` | Frames gesture must be held before firing |
| `COMMAND_COOLDOWN` | `1.2 s` | Minimum time between commands |
| `MP_DETECTION_CONFIDENCE` | `0.7` | MediaPipe detection threshold |
| `PATROL_ROUTE` | square 0.5 m | List of (dx,dy,dz,dyaw,dur) steps |

---

## File Structure

```
handfly/
├── main.py              # Entry point (real hardware)
├── run_mock.py          # Entry point (webcam mock)
├── config.py            # All tunable parameters
├── gestures.py          # Gesture enum definitions
├── gesture_detector.py  # MediaPipe + temporal buffer
├── flight_controller.py # Drone state machine + cflib
├── aideck_stream.py     # TCP frame reader + webcam mock
├── overlay.py           # cv2 annotation renderer
└── requirements.txt
```

---

## Safety Notes

- Always have a **manual override** (physical kill switch or human spotter) ready.
- The `FIST` gesture immediately halts movement — practise it first.
- `COMMAND_COOLDOWN` and `GESTURE_HOLD_FRAMES` prevent accidental triggers; increase them if the environment is noisy.
- The controller enforces a **floor at z = 0.2 m** to prevent ground collisions.
- Run in an open indoor space within the **lighthouse / flow deck tracking volume**.
