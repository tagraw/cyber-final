"""
Gesture definitions shared across the pipeline.
"""
from enum import Enum, auto


class Gesture(Enum):
    # Directional
    POINT_UP      = auto()   # index finger up  → ascend / move up
    POINT_DOWN    = auto()   # index finger down → descend / move down
    POINT_LEFT    = auto()   # index finger left → move left
    POINT_RIGHT   = auto()   # index finger right → move right
    POINT_FORWARD = auto()   # index + middle forward (camera axis) → move forward
    POINT_BACK    = auto()   # palm facing camera → move backward

    # Hold / toggle
    FIST          = auto()   # closed fist → hold position
    OPEN_PALM     = auto()   # flat open hand → takeoff / land toggle

    # Rotation
    SPIN_CW       = auto()   # circular CW motion → 360° yaw spin CW
    SPIN_CCW      = auto()   # circular CCW motion → 360° yaw spin CCW

    # Speed
    PEACE         = auto()   # V / peace sign → toggle speed mode

    # Altitude
    THUMBS_UP     = auto()   # → ascend
    THUMBS_DOWN   = auto()   # → descend

    # Preset routine
    PINCH         = auto()   # thumb-to-index → run patrol route

    UNKNOWN       = auto()
