"""
run_mock.py
-----------
Development entry point: runs the full HandFly CV + gesture pipeline
using your laptop webcam. No drone, no AIDeck, no cflib required.

Run from the handfly/ directory:
    python run_mock.py

Press Q to quit.
"""

import sys
import time
import threading
from enum import Enum, auto
from dataclasses import dataclass

# ── Verify cv2 opens before anything else ─────────────────────────────────
import cv2
import numpy as np

print("=== HandFly MOCK MODE ===")
print(f"OpenCV  : {cv2.__version__}")

try:
    import mediapipe as mp
    print(f"MediaPipe: {mp.__version__}")
except ImportError:
    print("ERROR: mediapipe not installed.  Run:  pip install mediapipe")
    sys.exit(1)

from config import Config
from gestures import Gesture
from gesture_detector import GestureDetector
from overlay import Overlay



# ---------------------------------------------------------------------------
# Minimal DroneState enum — no cflib import needed
# ---------------------------------------------------------------------------

class DroneState(Enum):
    GROUNDED   = auto()
    TAKING_OFF = auto()
    FLYING     = auto()
    HOVERING   = auto()
    LANDING    = auto()
    PATROL     = auto()
    SPINNING   = auto()


# ---------------------------------------------------------------------------
# Mock position tracker
# ---------------------------------------------------------------------------

@dataclass
class MockPosition:
    x:   float = 0.0
    y:   float = 0.0
    z:   float = 0.0
    yaw: float = 0.0


# ---------------------------------------------------------------------------
# Simulated flight controller
# ---------------------------------------------------------------------------

class MockFlightController:
    """Mirrors FlightController's public API; prints instead of flying."""

    def __init__(self):
        self.state = DroneState.GROUNDED
        self._pos  = MockPosition()
        self._speed_label = "SLOW"
        self._lock = threading.Lock()

    def handle_gesture(self, gesture: Gesture):
        with self._lock:
            dispatch = {
                Gesture.OPEN_PALM:     self._toggle_flight,
                Gesture.FIST:          lambda: self._log("HOLD position"),
                Gesture.POINT_UP:      lambda: self._move(dz= Config.STEP_Z),
                Gesture.POINT_DOWN:    lambda: self._move(dz=-Config.STEP_Z),
                Gesture.POINT_LEFT:    lambda: self._move(dy= Config.STEP_XY),
                Gesture.POINT_RIGHT:   lambda: self._move(dy=-Config.STEP_XY),
                Gesture.POINT_FORWARD: lambda: self._move(dx= Config.STEP_XY),
                Gesture.POINT_BACK:    lambda: self._move(dx=-Config.STEP_XY),
                Gesture.THUMBS_UP:     lambda: self._move(dz= Config.STEP_Z),
                Gesture.THUMBS_DOWN:   lambda: self._move(dz=-Config.STEP_Z),
                Gesture.SPIN_CW:       lambda: self._log("SPIN CW 360deg"),
                Gesture.SPIN_CCW:      lambda: self._log("SPIN CCW 360deg"),
                Gesture.PEACE:         self._toggle_speed,
                Gesture.PINCH:         self._start_patrol,
            }
            action = dispatch.get(gesture)
            if action:
                action()

    def emergency_land(self):
        self._log("EMERGENCY LAND")
        self.state = DroneState.GROUNDED

    # ── Actions ─────────────────────────────────────────────────────────

    def _toggle_flight(self):
        if self.state == DroneState.GROUNDED:
            self.state = DroneState.FLYING
            self._log("TAKEOFF")
        elif self.state in (DroneState.FLYING, DroneState.HOVERING):
            self.state = DroneState.GROUNDED
            self._log("LAND")

    def _move(self, dx=0.0, dy=0.0, dz=0.0):
        if self.state not in (DroneState.FLYING, DroneState.HOVERING):
            self._log("(not airborne - open palm to take off first)")
            return
        self._pos.x += dx
        self._pos.y += dy
        self._pos.z  = max(0.0, self._pos.z + dz)
        self._log(f"GO_TO  x={self._pos.x:+.2f}  y={self._pos.y:+.2f}  "
                  f"z={self._pos.z:.2f}")

    def _toggle_speed(self):
        self._speed_label = "FAST" if self._speed_label == "SLOW" else "SLOW"
        self._log(f"SPEED -> {self._speed_label}")

    def _start_patrol(self):
        if self.state not in (DroneState.FLYING, DroneState.HOVERING):
            return
        self.state = DroneState.PATROL
        self._log("PATROL START")
        threading.Thread(target=self._run_patrol, daemon=True).start()

    def _run_patrol(self):
        for i, step in enumerate(Config.PATROL_ROUTE):
            time.sleep(step[4] + 0.3)
            with self._lock:
                if self.state != DroneState.PATROL:
                    return
                self._log(f"PATROL waypoint {i+1}/{len(Config.PATROL_ROUTE)}")
        with self._lock:
            if self.state == DroneState.PATROL:
                self.state = DroneState.FLYING
                self._log("PATROL COMPLETE")

    @staticmethod
    def _log(msg: str):
        print(f"[MockDrone] {msg}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    # ── Open webcam ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0). Try index 1 or 2.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Drain a few frames so camera exposure settles
    for _ in range(5):
        cap.read()

    print("\nWebcam opened OK.")
    print("Controls: Q = quit | gestures printed to console")
    print("Open palm -> takeoff/land | Fist -> hold | Point -> move\n")

    cv2.namedWindow(Config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    detector = GestureDetector()
    ctrl     = MockFlightController()
    overlay  = Overlay()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed - exiting.")
            break

        # ── Gesture recognition ────────────────────────────────────────
        result = detector.process(frame)

        if result.gesture is not None:
            ctrl.handle_gesture(result.gesture)

        # ── Draw overlay ───────────────────────────────────────────────
        annotated = overlay.draw(frame, result, ctrl.state)

        # ── Show ───────────────────────────────────────────────────────
        cv2.imshow(Config.WINDOW_NAME, annotated)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            print("Quit.")
            break

    ctrl.emergency_land()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()