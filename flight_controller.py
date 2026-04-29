"""
FlightController
----------------
Receives confirmed Gesture values and issues cflib HighLevelCommander calls.

State machine
~~~~~~~~~~~~~
  GROUNDED → (OPEN_PALM) → FLYING
  FLYING   → (OPEN_PALM) → LANDING → GROUNDED
  FLYING   → (PINCH)     → PATROL (autonomous) → FLYING
  Any state: FIST → hold position
"""

import threading
import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

from cflib.crazyflie.high_level_commander import HighLevelCommander

from config import Config
from gestures import Gesture

log = logging.getLogger(__name__)


class DroneState(Enum):
    GROUNDED  = auto()
    TAKING_OFF = auto()
    FLYING    = auto()
    HOVERING  = auto()
    LANDING   = auto()
    PATROL    = auto()
    SPINNING  = auto()


@dataclass
class DronePosition:
    x:   float = 0.0
    y:   float = 0.0
    z:   float = Config.DEFAULT_HEIGHT
    yaw: float = 0.0


class FlightController:
    def __init__(self, scf):
        self._scf = scf
        self._hlc: Optional[HighLevelCommander] = None
        self._pos = DronePosition()
        self.state = DroneState.GROUNDED
        self._speed = Config.SPEED_SLOW
        self._lock = threading.Lock()
        self._patrol_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_gesture(self, gesture: Gesture):
        """Dispatch gesture → drone action (non-blocking where possible)."""
        with self._lock:
            log.debug("Gesture: %s  State: %s", gesture, self.state)

            # Patrol is autonomous – only FIST can interrupt it
            if self.state == DroneState.PATROL:
                if gesture == Gesture.FIST:
                    self._abort_patrol()
                return

            handler = {
                Gesture.OPEN_PALM:   self._toggle_flight,
                Gesture.FIST:        self._hover,
                Gesture.POINT_UP:    lambda: self._translate(0, 0,  Config.STEP_Z),
                Gesture.POINT_DOWN:  lambda: self._translate(0, 0, -Config.STEP_Z),
                Gesture.POINT_LEFT:  lambda: self._translate(0,  Config.STEP_XY, 0),
                Gesture.POINT_RIGHT: lambda: self._translate(0, -Config.STEP_XY, 0),
                Gesture.POINT_FORWARD: lambda: self._translate( Config.STEP_XY, 0, 0),
                Gesture.POINT_BACK:  lambda: self._translate(-Config.STEP_XY, 0, 0),
                Gesture.THUMBS_UP:   lambda: self._translate(0, 0,  Config.STEP_Z),
                Gesture.THUMBS_DOWN: lambda: self._translate(0, 0, -Config.STEP_Z),
                Gesture.PEACE:       self._toggle_speed,
                Gesture.SPIN_CW:     lambda: self._spin(Config.SPIN_DEGREES),
                Gesture.SPIN_CCW:    lambda: self._spin(-Config.SPIN_DEGREES),
                Gesture.PINCH:       self._start_patrol,
            }

            action = handler.get(gesture)
            if action:
                action()

    def emergency_land(self):
        """Called on shutdown regardless of state."""
        try:
            if self._hlc and self.state not in (DroneState.GROUNDED,
                                                DroneState.LANDING):
                self._hlc.land(0.0, Config.LAND_DURATION)
                time.sleep(Config.LAND_DURATION + 0.5)
                self._hlc.stop()
        except Exception as e:
            log.error("Emergency land error: %s", e)

    # ------------------------------------------------------------------
    # Internal actions
    # ------------------------------------------------------------------

    def _toggle_flight(self):
        if self.state == DroneState.GROUNDED:
            self._takeoff()
        elif self.state in (DroneState.FLYING, DroneState.HOVERING):
            self._land()

    def _takeoff(self):
        print("[Flight] Taking off...")
        self._hlc = HighLevelCommander(self._scf.cf)
        self.state = DroneState.TAKING_OFF
        self._pos = DronePosition()
        self._hlc.takeoff(Config.DEFAULT_HEIGHT, Config.TAKEOFF_DURATION)
        # Wait asynchronously so the lock isn't held
        threading.Thread(target=self._finish_takeoff, daemon=True).start()

    def _finish_takeoff(self):
        time.sleep(Config.TAKEOFF_DURATION + 0.5)
        with self._lock:
            self.state = DroneState.FLYING
            print("[Flight] Airborne – gesture control active.")

    def _land(self):
        print("[Flight] Landing...")
        self.state = DroneState.LANDING
        self._hlc.land(0.0, Config.LAND_DURATION)
        threading.Thread(target=self._finish_land, daemon=True).start()

    def _finish_land(self):
        time.sleep(Config.LAND_DURATION + 0.5)
        with self._lock:
            self._hlc.stop()
            self._hlc = None
            self.state = DroneState.GROUNDED
            print("[Flight] Landed.")

    def _hover(self):
        if self.state not in (DroneState.FLYING, DroneState.HOVERING):
            return
        print("[Flight] Holding position.")
        self.state = DroneState.HOVERING
        self._hlc.go_to(
            self._pos.x, self._pos.y, self._pos.z, self._pos.yaw,
            0.5, relative=False
        )

    def _translate(self, dx: float, dy: float, dz: float):
        if self.state not in (DroneState.FLYING, DroneState.HOVERING):
            return
        self._pos.x += dx
        self._pos.y += dy
        self._pos.z = max(0.2, self._pos.z + dz)  # floor safety
        print(f"[Flight] Move → x={self._pos.x:.2f} y={self._pos.y:.2f} "
              f"z={self._pos.z:.2f}")
        self.state = DroneState.FLYING
        self._hlc.go_to(
            self._pos.x, self._pos.y, self._pos.z, self._pos.yaw,
            Config.MOVE_DURATION, relative=False
        )

    def _spin(self, degrees: float):
        if self.state not in (DroneState.FLYING, DroneState.HOVERING):
            return
        direction = "CW" if degrees > 0 else "CCW"
        print(f"[Flight] Spinning {direction} 360°")
        self.state = DroneState.SPINNING
        target_yaw = self._pos.yaw + degrees
        self._hlc.go_to(
            self._pos.x, self._pos.y, self._pos.z, target_yaw,
            2.5, relative=False
        )
        self._pos.yaw = target_yaw % 360
        threading.Thread(target=self._finish_spin, daemon=True).start()

    def _finish_spin(self):
        time.sleep(3.0)
        with self._lock:
            if self.state == DroneState.SPINNING:
                self.state = DroneState.FLYING

    def _toggle_speed(self):
        if self._speed == Config.SPEED_SLOW:
            self._speed = Config.SPEED_FAST
            print("[Flight] Speed → FAST")
        else:
            self._speed = Config.SPEED_SLOW
            print("[Flight] Speed → SLOW")

    # ------------------------------------------------------------------
    # Patrol routine
    # ------------------------------------------------------------------

    def _start_patrol(self):
        if self.state not in (DroneState.FLYING, DroneState.HOVERING):
            return
        print("[Flight] Starting patrol route...")
        self.state = DroneState.PATROL
        self._patrol_thread = threading.Thread(
            target=self._run_patrol, daemon=True
        )
        self._patrol_thread.start()

    def _run_patrol(self):
        for (dx, dy, dz, dyaw, dur) in Config.PATROL_ROUTE:
            with self._lock:
                if self.state != DroneState.PATROL:
                    break
                self._pos.x   += dx
                self._pos.y   += dy
                self._pos.z    = max(0.2, self._pos.z + dz)
                self._pos.yaw += dyaw
                print(f"[Patrol] → ({self._pos.x:.2f}, {self._pos.y:.2f}, "
                      f"{self._pos.z:.2f})")
                self._hlc.go_to(
                    self._pos.x, self._pos.y, self._pos.z, self._pos.yaw,
                    dur, relative=False
                )
            time.sleep(dur + 0.3)

        with self._lock:
            if self.state == DroneState.PATROL:
                self.state = DroneState.FLYING
                print("[Flight] Patrol complete.")

    def _abort_patrol(self):
        print("[Flight] Patrol aborted – holding position.")
        self.state = DroneState.HOVERING
        self._hlc.go_to(
            self._pos.x, self._pos.y, self._pos.z, self._pos.yaw,
            0.5, relative=False
        )
