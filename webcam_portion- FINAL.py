import cv2
import mediapipe as mp
import time
import math
from enum import Enum, auto

# ================= ENUM =================
class Gesture(Enum):
    NONE = auto()
    FIST = auto()
    OPEN_PALM = auto()
    TWO_FINGERS = auto()
    THREE_FINGERS = auto()
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    PINCH = auto()

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= STATE =================
last_gesture = Gesture.NONE
gesture_count = 0
CONFIRM_FRAMES = 8

last_printed = Gesture.NONE
prev_time = 0

# ================= UTILS =================
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

# ================= ROBUST FINGER DETECTION =================

def finger_extended(lm, tip, pip):
    """Tip above its own PIP joint = extended (standard approach)"""
    return lm[tip].y < lm[pip].y

def thumb_extended(lm):
    return lm[4].x < lm[3].x  # for right hand; tip left of knuckle

def count_fingers(lm):
    # tip landmark, pip landmark pairs
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    return sum(1 for tip, pip in finger_pairs if finger_extended(lm, tip, pip))

def which_fingers_up(lm):
    """Returns list of which fingers are extended: index=0, middle=1, ring=2, pinky=3"""
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    return [i for i, (tip, pip) in enumerate(finger_pairs) if finger_extended(lm, tip, pip)]

# ================= BASIC GESTURES =================

def is_fist(lm):
    return count_fingers(lm) == 0

def is_open_palm(lm):
    return count_fingers(lm) == 4

def is_pinch(lm):
    return dist(lm[4], lm[8]) < 0.05

def is_pointing(lm):
    """Index finger must be up; ring and pinky must be curled"""
    up = which_fingers_up(lm)
    # Index (0) must be extended, ring (2) and pinky (3) must be curled
    # Middle (1) is allowed to be up — it often lifts when pointing down
    index_up = 0 in up
    middle_curled = 1 not in up

    ring_curled = 2 not in up
    pinky_curled = 3 not in up
    return index_up and middle_curled and ring_curled and pinky_curled

def pointing_direction(lm):
    """Determine direction based on index finger tip vs MCP (knuckle) angle"""
    tip = lm[8]
    mcp = lm[5]  # index finger base knuckle

    dx = tip.x - mcp.x
    dy = tip.y - mcp.y  # y increases downward in image coords

    if abs(dx) > abs(dy):
        return Gesture.RIGHT if dx > 0 else Gesture.LEFT
    else:
        return Gesture.DOWN if dy > 0 else Gesture.UP

# ================= MAIN CLASSIFIER =================

def recognize_static(lm):
    # Priority order matters here

    if is_fist(lm):
        return Gesture.FIST

    if is_pinch(lm):
        return Gesture.PINCH

    if is_pointing(lm):
        return pointing_direction(lm)

    fingers = count_fingers(lm)

    if fingers == 2:
        return Gesture.TWO_FINGERS

    if fingers == 3:
        return Gesture.THREE_FINGERS

    if fingers == 4:
        return Gesture.OPEN_PALM

    return Gesture.NONE
# ================= STABILITY FILTER =================

def stable_gesture(g):
    global last_gesture, gesture_count

    if g == last_gesture:
        gesture_count += 1
    else:
        last_gesture = g
        gesture_count = 0

    if gesture_count >= CONFIRM_FRAMES:
        return g
    return None

# ================= MAIN =================

try:
    print("Starting HandFly Finger System (FIXED)...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Camera not found")
        exit()

    print("Press ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        detected = Gesture.NONE

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark

            mp_draw.draw_landmarks(
                frame,
                result.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS
            )

            detected = recognize_static(lm)
            stable = stable_gesture(detected)

            if stable and stable != Gesture.NONE:
                if stable != last_printed:
                    print("GESTURE:", stable)
                    last_printed = stable

                cv2.putText(frame, stable.name,
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

            # 🔍 DEBUG: finger count
            cv2.putText(frame, f"Fingers: {count_fingers(lm)}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2)

        else:
            last_gesture = Gesture.NONE
            gesture_count = 0

        # FPS display
        cv2.putText(frame, f"FPS: {int(fps)}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2)

        cv2.imshow("HandFly Webcam", frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

except Exception as e:
    print("ERROR:", e)

finally:
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()