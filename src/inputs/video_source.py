"""Video input source adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import cv2


class VideoSourceError(Exception):
    """Raised when a video source cannot be opened or read."""


class VideoSource:
    """Read and stream frames from video files."""

    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._path: Path | None = None

    def open(self, video_path: str | Path) -> None:
        """Open a video file for frame-by-frame reading."""
        path = Path(video_path)
        if not path.exists():
            raise VideoSourceError(f"Video file not found: {path}")

        self.release()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"Failed to open video file: {path}")

        self._capture = capture
        self._path = path

    def read(self) -> tuple[bool, Any]:
        """Read one frame from opened video."""
        if self._capture is None:
            raise VideoSourceError("Video source is not opened. Call open() first.")
        success, frame = self._capture.read()
        return success, frame

    def frames(self) -> Generator[Any, None, None]:
        """Yield all frames until stream ends."""
        if self._capture is None:
            raise VideoSourceError("Video source is not opened. Call open() first.")

        while True:
            success, frame = self._capture.read()
            if not success:
                break
            yield frame

    def metadata(self) -> dict[str, Any]:
        """Return common video metadata for UI and logs."""
        if self._capture is None:
            raise VideoSourceError("Video source is not opened. Call open() first.")

        return {
            "path": str(self._path) if self._path else "",
            "fps": float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0),
            "frame_count": int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }

    def release(self) -> None:
        """Release video capture resources."""
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._path = None
