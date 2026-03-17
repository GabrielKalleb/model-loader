"""Image input source adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


class ImageSourceError(Exception):
    """Raised when an image input cannot be read."""


class ImageSource:
    """Load images for offline inference."""

    def read(self, image_path: str | Path) -> Any:
        """Read one image from disk and return it as ndarray (BGR)."""
        path = Path(image_path)
        if not path.exists():
            raise ImageSourceError(f"Image file not found: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ImageSourceError(f"Failed to decode image: {path}")
        return image

    def read_batch(
        self,
        image_dir: str | Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ) -> list[tuple[Path, Any]]:
        """Read all supported images from a directory."""
        directory = Path(image_dir)
        if not directory.exists() or not directory.is_dir():
            raise ImageSourceError(f"Image directory not found: {directory}")

        result: list[tuple[Path, Any]] = []
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            result.append((path, image))

        if not result:
            raise ImageSourceError(f"No readable images found in: {directory}")
        return result
