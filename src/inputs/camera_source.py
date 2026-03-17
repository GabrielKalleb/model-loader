"""Camera input source adapter."""

from __future__ import annotations

from typing import Any, Generator

import cv2


class CameraSourceError(Exception):
    """Raised when camera stream cannot be opened or read."""


class CameraSource:
    """Read real-time frames from a camera device."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._capture: cv2.VideoCapture | None = None

    def start(self, width: int | None = None, height: int | None = None, fps: int | None = None) -> None:
        """Open camera stream and optionally set capture properties."""
        if self._capture is not None:
            return

        capture = cv2.VideoCapture(self.device_index)
        if not capture.isOpened():
            capture.release()
            raise CameraSourceError(f"Failed to open camera device: {self.device_index}")

        if width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps is not None:
            capture.set(cv2.CAP_PROP_FPS, float(fps))

        self._capture = capture

    def read(self) -> tuple[bool, Any]:
        """Read one frame from camera."""
        if self._capture is None:
            raise CameraSourceError("Camera is not started. Call start() first.")
        return self._capture.read()

    def frames(self) -> Generator[Any, None, None]:
        """Yield frames from camera stream until a read fails."""
        if self._capture is None:
            raise CameraSourceError("Camera is not started. Call start() first.")

        while True:
            success, frame = self._capture.read()
            if not success:
                break
            yield frame

    def metadata(self) -> dict[str, Any]:
        """Return runtime camera stream metadata."""
        if self._capture is None:
            raise CameraSourceError("Camera is not started. Call start() first.")

        return {
            "device_index": self.device_index,
            "fps": float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }

    def stop(self) -> None:
        """Release camera stream resources."""
        if self._capture is not None:
            self._capture.release()
        self._capture = None
