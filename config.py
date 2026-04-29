"""
HandFly Configuration
All tunable parameters in one place.
"""


class Config:
    # ── Drone ──────────────────────────────────────────────────────────────
    URI = "radio://0/80/2M/E7E7E7E701"

    DEFAULT_HEIGHT   = 0.5    # metres
    STEP_XY          = 0.3    # metres per directional command
    STEP_Z           = 0.2    # metres per ascend/descend command
    YAW_STEP         = 45.0   # degrees per yaw command
    SPIN_DEGREES     = 360.0  # full spin
    MOVE_DURATION    = 1.0    # seconds for go_to calls
    TAKEOFF_DURATION = 2.0
    LAND_DURATION    = 2.0

    SPEED_SLOW = 0.3   # m/s
    SPEED_FAST = 0.8   # m/s

    # ── AIDeck Stream ──────────────────────────────────────────────────────
    AIDECK_IP   = "192.168.4.1"   # default AIDeck Wi-Fi IP
    AIDECK_PORT = 5000

    # ── Gesture Recognition ────────────────────────────────────────────────
    # How many consecutive frames a gesture must be seen before firing
    GESTURE_HOLD_FRAMES = 8

    # MediaPipe confidence thresholds
    MP_DETECTION_CONFIDENCE  = 0.7
    MP_TRACKING_CONFIDENCE   = 0.6

    # Cooldown between consecutive commands (seconds)
    COMMAND_COOLDOWN = 1.2

    # ── Preset Patrol Route (square) ───────────────────────────────────────
    # List of (dx, dy, dz, dyaw, duration) relative moves
    PATROL_ROUTE = [
        ( 0.5,  0.0, 0.0,  0.0, 1.5),
        ( 0.0,  0.5, 0.0,  0.0, 1.5),
        (-0.5,  0.0, 0.0,  0.0, 1.5),
        ( 0.0, -0.5, 0.0,  0.0, 1.5),
    ]

    # ── Display ────────────────────────────────────────────────────────────
    WINDOW_NAME  = "HandFly – AIDeck Feed"
    FRAME_WIDTH  = 324
    FRAME_HEIGHT = 244
