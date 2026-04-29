"""
HandFly - Gesture-controlled Crazyflie drone via AIDeck + MediaPipe
Entry point: launches the CV pipeline and drone controller together.
"""

import logging
import threading
import time
import sys

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from config import Config
from gesture_detector import GestureDetector
from flight_controller import FlightController
from aideck_stream import AIDeckStream
from overlay import Overlay

logging.basicConfig(level=logging.ERROR)


def main():
    print("=== HandFly: Gesture-Controlled Crazyflie ===")
    print(f"Connecting to drone at {Config.URI} ...")

    cflib.crtp.init_drivers()

    with SyncCrazyflie(Config.URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        print("Connected!")

        flight_ctrl = FlightController(scf)
        gesture_detector = GestureDetector()
        overlay = Overlay()

        # AIDeck video stream runs in a background thread
        stream = AIDeckStream(
            host=Config.AIDECK_IP,
            port=Config.AIDECK_PORT,
        )
        stream_thread = threading.Thread(target=stream.run, daemon=True)
        stream_thread.start()

        print("AIDeck stream started. Show gestures to control the drone.")
        print("Press 'q' in the video window to quit.\n")

        try:
            while True:
                frame = stream.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # --- Computer Vision ---
                result = gesture_detector.process(frame)

                # --- Flight Control ---
                if result.gesture is not None:
                    flight_ctrl.handle_gesture(result.gesture)

                # --- Overlay & Display ---
                annotated = overlay.draw(frame, result, flight_ctrl.state)
                if overlay.show(annotated):   # returns True when 'q' pressed
                    break

        except KeyboardInterrupt:
            print("\nInterrupt received – landing...")
        finally:
            flight_ctrl.emergency_land()
            stream.stop()
            print("Shutdown complete.")


if __name__ == "__main__":
    main()
