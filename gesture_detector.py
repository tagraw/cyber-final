"""
GestureDetector
---------------
Wraps MediaPipe Hands and classifies hand poses into Gesture enum values.

Detection strategy
~~~~~~~~~~~~~~~~~~
1. Run MediaPipe to get 21 3D landmarks per hand.
2. Compute per-finger extension flags (tip above pip joint in image space).
3. Compute Euclidean distances for pinch / thumbs logic.
4. Apply a temporal buffer (GESTURE_HOLD_FRAMES) before emitting a command.
5. Maintain a circular-motion tracker for spin gestures.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import cv2
import mediapipe as mp
import numpy as np

from config import Config
from gestures import Gesture


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HandResult:
    """Everything the rest of the pipeline needs from one processed frame."""
    gesture: Optional[Gesture]          # confirmed gesture (after hold timer)
    raw_gesture: Optional[Gesture]      # current unconfirmed pose
    landmarks: Optional[object]         # mediapipe NormalizedLandmarkList
    handedness: Optional[str]           # "Left" or "Right"
    confidence: float = 0.0
    wrist_trail: List[Tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe hand topology)
# ---------------------------------------------------------------------------
WRIST       = 0
THUMB_CMC   = 1;  THUMB_MCP  = 2;  THUMB_IP  = 3;  THUMB_TIP  = 4
INDEX_MCP   = 5;  INDEX_PIP  = 6;  INDEX_DIP = 7;  INDEX_TIP  = 8
MIDDLE_MCP  = 9;  MIDDLE_PIP = 10; MIDDLE_DIP= 11; MIDDLE_TIP = 12
RING_MCP    = 13; RING_PIP   = 14; RING_DIP  = 15; RING_TIP   = 16
PINKY_MCP   = 17; PINKY_PIP  = 18; PINKY_DIP = 19; PINKY_TIP  = 20


# ---------------------------------------------------------------------------
# Helper geometry
# ---------------------------------------------------------------------------

def _dist(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _finger_extended(lm, tip: int, pip: int) -> bool:
    """True when fingertip is above (lower y in image) its PIP joint."""
    return lm[tip].y < lm[pip].y


def _thumb_extended(lm, handedness: str) -> bool:
    """Thumb uses x-axis comparison (flipped for left vs right hand)."""
    if handedness == "Right":
        return lm[THUMB_TIP].x < lm[THUMB_IP].x
    else:
        return lm[THUMB_TIP].x > lm[THUMB_IP].x


# ---------------------------------------------------------------------------
# Circular-motion tracker (for spin gestures)
# ---------------------------------------------------------------------------

class CircleTracker:
    """
    Accumulates wrist positions and tries to detect a circular arc.
    Returns 'CW', 'CCW', or None.
    """
    TRAIL_LEN   = 30   # frames
    MIN_RADIUS  = 0.08  # normalised units
    ARC_THRESH  = 300   # degrees of angular coverage needed

    def __init__(self):
        self._trail: deque = deque(maxlen=self.TRAIL_LEN)

    def update(self, x: float, y: float) -> Optional[str]:
        self._trail.append((x, y))
        if len(self._trail) < self.TRAIL_LEN:
            return None

        pts = np.array(self._trail)
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        radii = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        if radii.mean() < self.MIN_RADIUS:
            return None

        # Accumulate signed angle increments
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        diffs = np.diff(np.unwrap(angles)) * (180 / math.pi)
        total = diffs.sum()

        if abs(total) >= self.ARC_THRESH:
            self._trail.clear()
            return "CW" if total > 0 else "CCW"
        return None

    def reset(self):
        self._trail.clear()


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class GestureDetector:
    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=Config.MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MP_TRACKING_CONFIDENCE,
        )

        # Temporal buffer
        self._hold_buffer: deque = deque(maxlen=Config.GESTURE_HOLD_FRAMES)
        self._last_confirmed: Optional[Gesture] = None
        self._last_command_time: float = 0.0

        # Circular motion
        self._circle_tracker = CircleTracker()
        self._wrist_trail: deque = deque(maxlen=30)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> HandResult:
        """
        Process one BGR frame.
        Returns a HandResult with the confirmed gesture (or None if cooling
        down / not enough hold frames yet).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = self._hands.process(rgb)

        if not mp_result.multi_hand_landmarks:
            self._hold_buffer.clear()
            self._circle_tracker.reset()
            self._wrist_trail.clear()
            return HandResult(gesture=None, raw_gesture=None,
                              landmarks=None, handedness=None)

        lm = mp_result.multi_hand_landmarks[0].landmark
        handedness = (mp_result.multi_handedness[0]
                      .classification[0].label)
        confidence = (mp_result.multi_handedness[0]
                      .classification[0].score)

        # Wrist trail for circle detection
        wx, wy = lm[WRIST].x, lm[WRIST].y
        self._wrist_trail.append((wx, wy))
        circle_dir = self._circle_tracker.update(wx, wy)

        raw = self._classify(lm, handedness, circle_dir)
        self._hold_buffer.append(raw)

        confirmed = self._confirm()
        return HandResult(
            gesture=confirmed,
            raw_gesture=raw,
            landmarks=mp_result.multi_hand_landmarks[0],
            handedness=handedness,
            confidence=confidence,
            wrist_trail=list(self._wrist_trail),
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self, lm, handedness: str,
                  circle_dir: Optional[str]) -> Gesture:
        # Finger extension flags
        idx_ext  = _finger_extended(lm, INDEX_TIP,  INDEX_PIP)
        mid_ext  = _finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP)
        ring_ext = _finger_extended(lm, RING_TIP,   RING_PIP)
        pink_ext = _finger_extended(lm, PINKY_TIP,  PINKY_PIP)
        thm_ext  = _thumb_extended(lm, handedness)

        num_fingers = sum([idx_ext, mid_ext, ring_ext, pink_ext])

        # ── Pinch (thumb tip close to index tip) ───────────────────────
        pinch_dist = _dist(lm[THUMB_TIP], lm[INDEX_TIP])
        if pinch_dist < 0.05 and not mid_ext:
            return Gesture.PINCH

        # ── Fist ───────────────────────────────────────────────────────
        if num_fingers == 0 and not thm_ext:
            return Gesture.FIST

        # ── Open palm ─────────────────────────────────────────────────
        if num_fingers == 4 and thm_ext:
            return Gesture.OPEN_PALM

        # ── Peace / V sign ────────────────────────────────────────────
        if idx_ext and mid_ext and not ring_ext and not pink_ext:
            return Gesture.PEACE

        # ── Thumbs up ─────────────────────────────────────────────────
        if thm_ext and num_fingers == 0:
            if lm[THUMB_TIP].y < lm[WRIST].y:
                return Gesture.THUMBS_UP

        # ── Thumbs down ───────────────────────────────────────────────
        if thm_ext and num_fingers == 0:
            if lm[THUMB_TIP].y > lm[WRIST].y:
                return Gesture.THUMBS_DOWN

        # ── Single index finger pointing ───────────────────────────────
        if idx_ext and not mid_ext and not ring_ext and not pink_ext:
            return self._pointing_direction(lm)

        # ── Circular spin gesture ──────────────────────────────────────
        if circle_dir == "CW":
            return Gesture.SPIN_CW
        if circle_dir == "CCW":
            return Gesture.SPIN_CCW

        return Gesture.UNKNOWN

    def _pointing_direction(self, lm) -> Gesture:
        """Determine which way the index finger is pointing."""
        dx = lm[INDEX_TIP].x - lm[INDEX_MCP].x
        dy = lm[INDEX_TIP].y - lm[INDEX_MCP].y   # y increases downward
        dz = lm[INDEX_TIP].z - lm[INDEX_MCP].z   # z negative = toward camera

        # Dominant axis
        abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)

        if abs_dy >= abs_dx and abs_dy >= abs_dz:
            return Gesture.POINT_UP if dy < 0 else Gesture.POINT_DOWN
        if abs_dx >= abs_dy and abs_dx >= abs_dz:
            return Gesture.POINT_LEFT if dx < 0 else Gesture.POINT_RIGHT
        # z-axis dominant → toward / away from camera
        return Gesture.POINT_FORWARD if dz < 0 else Gesture.POINT_BACK

    # ------------------------------------------------------------------
    # Temporal confirmation
    # ------------------------------------------------------------------

    def _confirm(self) -> Optional[Gesture]:
        if len(self._hold_buffer) < Config.GESTURE_HOLD_FRAMES:
            return None

        # All frames in buffer must agree
        if len(set(self._hold_buffer)) != 1:
            return None

        candidate = self._hold_buffer[-1]
        if candidate == Gesture.UNKNOWN:
            return None

        # Command cooldown
        now = time.time()
        if now - self._last_command_time < Config.COMMAND_COOLDOWN:
            return None

        # Don't repeat the same gesture continuously (except FIST = hover)
        if candidate == self._last_confirmed and candidate != Gesture.FIST:
            # Require buffer to have been cleared since last confirmation
            return None

        self._last_confirmed = candidate
        self._last_command_time = now
        self._hold_buffer.clear()
        return candidate
