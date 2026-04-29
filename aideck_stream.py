"""
AIDeckStream
------------
Receives JPEG frames from the AIDeck Wi-Fi hotspot over a raw TCP socket
(the standard AIDeck CPX/streamer firmware protocol).

Protocol:
  Each frame is preceded by a 4-byte little-endian length header, followed by
  that many bytes of JPEG data.  The stream runs continuously; we just
  decode the latest frame and make it available via get_frame().

If you are running the system WITHOUT an AIDeck (e.g. for desktop testing),
set AIDECK_MOCK=True in config to use the host webcam instead.
"""

import socket
import struct
import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import Config


class AIDeckStream:
    """
    Background thread that keeps the latest frame in a buffer.
    """

    def __init__(self, host: str = Config.AIDECK_IP,
                 port: int = Config.AIDECK_PORT,
                 mock: bool = False):
        self._host = host
        self._port = port
        self._mock = mock
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        if self._mock:
            self._run_webcam()
        else:
            self._run_aideck()

    def stop(self):
        self._running = False

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    # ------------------------------------------------------------------
    # Real AIDeck stream
    # ------------------------------------------------------------------

    def _run_aideck(self):
        while self._running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self._host, self._port))
                sock.settimeout(5.0)
                print(f"[AIDeck] Connected to {self._host}:{self._port}")

                while self._running:
                    # Read 4-byte length header
                    raw_len = self._recv_exact(sock, 4)
                    if raw_len is None:
                        break
                    (length,) = struct.unpack("<I", raw_len)

                    # Read JPEG payload
                    payload = self._recv_exact(sock, length)
                    if payload is None:
                        break

                    # Decode
                    arr = np.frombuffer(payload, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        # Resize to expected dimensions
                        frame = cv2.resize(
                            frame,
                            (Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
                        )
                        with self._lock:
                            self._frame = frame

                sock.close()
            except (ConnectionRefusedError, OSError) as e:
                print(f"[AIDeck] Connection error: {e} – retrying in 2 s...")
                time.sleep(2.0)

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly n bytes or return None on disconnect."""
        buf = b""
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
            except socket.timeout:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    # ------------------------------------------------------------------
    # Mock mode: use host webcam (for development without hardware)
    # ------------------------------------------------------------------

    def _run_webcam(self):
        print("[AIDeck] MOCK MODE – using webcam")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

        while self._running:
            ret, frame = cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            time.sleep(1 / 30)

        cap.release()
