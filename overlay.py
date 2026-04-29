"""
Overlay
-------
Draws the MediaPipe hand skeleton, gesture label, drone state, and
speed mode onto each AIDeck frame, then shows it with cv2.imshow.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp

from config import Config
from gestures import Gesture
from flight_controller import DroneState


# Colour palette (BGR)
COL_GREEN  = (0,  220,  80)
COL_CYAN   = (0,  200, 200)
COL_RED    = (0,   50, 220)
COL_YELLOW = (0,  220, 220)
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (0,    0,   0)
COL_ORANGE = (0,  150, 255)

# Gesture → human-readable label
GESTURE_LABELS = {
    Gesture.POINT_UP:      "▲ UP",
    Gesture.POINT_DOWN:    "▼ DOWN",
    Gesture.POINT_LEFT:    "◀ LEFT",
    Gesture.POINT_RIGHT:   "▶ RIGHT",
    Gesture.POINT_FORWARD: "● FORWARD",
    Gesture.POINT_BACK:    "● BACK",
    Gesture.FIST:          "✊ HOLD",
    Gesture.OPEN_PALM:     "✋ TAKEOFF/LAND",
    Gesture.SPIN_CW:       "↻ SPIN CW",
    Gesture.SPIN_CCW:      "↺ SPIN CCW",
    Gesture.PEACE:         "✌ SPEED TOGGLE",
    Gesture.THUMBS_UP:     "👍 ASCEND",
    Gesture.THUMBS_DOWN:   "👎 DESCEND",
    Gesture.PINCH:         "🤌 PATROL",
    Gesture.UNKNOWN:       "",
}

STATE_COLOURS = {
    DroneState.GROUNDED:   COL_WHITE,
    DroneState.TAKING_OFF: COL_YELLOW,
    DroneState.FLYING:     COL_GREEN,
    DroneState.HOVERING:   COL_CYAN,
    DroneState.LANDING:    COL_ORANGE,
    DroneState.PATROL:     COL_CYAN,
    DroneState.SPINNING:   COL_YELLOW,
}

MP_DRAW = mp.solutions.drawing_utils
MP_STYLES = mp.solutions.drawing_styles
MP_HANDS = mp.solutions.hands


class Overlay:
    def __init__(self):
        self._prev_gesture: Optional[Gesture] = None
        self._gesture_display_frames = 0

    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray, result, drone_state: DroneState) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Hand skeleton ──────────────────────────────────────────────
        if result.landmarks is not None:
            MP_DRAW.draw_landmarks(
                out,
                result.landmarks,
                MP_HANDS.HAND_CONNECTIONS,
                MP_STYLES.get_default_hand_landmarks_style(),
                MP_STYLES.get_default_hand_connections_style(),
            )

            # Wrist trail
            trail = result.wrist_trail
            for i in range(1, len(trail)):
                x1 = int(trail[i-1][0] * w)
                y1 = int(trail[i-1][1] * h)
                x2 = int(trail[i][0]   * w)
                y2 = int(trail[i][1]   * h)
                alpha = i / len(trail)
                col = (int(80 * alpha), int(180 * alpha), int(255 * alpha))
                cv2.line(out, (x1, y1), (x2, y2), col, 1)

        # ── Gesture label ──────────────────────────────────────────────
        confirmed = result.gesture
        if confirmed is not None and confirmed != Gesture.UNKNOWN:
            self._prev_gesture = confirmed
            self._gesture_display_frames = 45

        label = ""
        if self._gesture_display_frames > 0 and self._prev_gesture:
            label = GESTURE_LABELS.get(self._prev_gesture, "")
            self._gesture_display_frames -= 1

        if label:
            _draw_text_with_bg(out, label, (10, h - 15),
                               font_scale=0.7, thickness=2,
                               fg=COL_GREEN, bg=COL_BLACK)

        # ── Raw pose (unconfirmed) ─────────────────────────────────────
        if result.raw_gesture and result.raw_gesture != Gesture.UNKNOWN:
            raw_txt = f"? {result.raw_gesture.name}"
            _draw_text_with_bg(out, raw_txt, (10, h - 40),
                               font_scale=0.45, thickness=1,
                               fg=COL_YELLOW, bg=COL_BLACK)

        # ── Drone state ────────────────────────────────────────────────
        state_col = STATE_COLOURS.get(drone_state, COL_WHITE)
        state_txt = f"State: {drone_state.name}"
        _draw_text_with_bg(out, state_txt, (10, 22),
                           font_scale=0.55, thickness=1,
                           fg=state_col, bg=COL_BLACK)

        # ── Confidence ────────────────────────────────────────────────
        if result.confidence > 0:
            conf_txt = f"Conf: {result.confidence:.0%}"
            _draw_text_with_bg(out, conf_txt, (10, 48),
                               font_scale=0.45, thickness=1,
                               fg=COL_WHITE, bg=COL_BLACK)

        # ── Handedness ────────────────────────────────────────────────
        if result.handedness:
            _draw_text_with_bg(out, result.handedness, (w - 70, 22),
                               font_scale=0.45, thickness=1,
                               fg=COL_CYAN, bg=COL_BLACK)

        return out

    def show(self, frame: np.ndarray) -> bool:
        """Display frame; return True if user pressed 'q'."""
        cv2.imshow(Config.WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        return key == ord("q")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _draw_text_with_bg(img, text: str, origin: Tuple[int, int],
                        font_scale: float = 0.6,
                        thickness: int = 1,
                        fg=(255, 255, 255),
                        bg=(0, 0, 0),
                        font=cv2.FONT_HERSHEY_SIMPLEX,
                        padding: int = 4):
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(img,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding),
                  bg, -1)
    cv2.putText(img, text, (x, y), font, font_scale, fg, thickness,
                cv2.LINE_AA)
