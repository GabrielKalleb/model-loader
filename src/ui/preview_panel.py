"""Inference preview panel."""

from __future__ import annotations

from typing import Any

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PreviewPanel(QWidget):
    """Render images, videos, and camera frames with detections."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_frame_shape: tuple[int, int] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel("Sem imagem")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 360)
        self.image_label.setStyleSheet("border: 1px solid #555;")
        layout.addWidget(self.image_label, stretch=1)

        self.info_label = QLabel("Pronto")
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.info_label)

    def update_frame(self, frame_bgr: Any, text: str | None = None) -> None:
        """Render BGR ndarray frame in preview label."""
        if frame_bgr is None:
            self.image_label.setText("Sem imagem")
            if text:
                self.info_label.setText(text)
            return

        if not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
            self.info_label.setText("Frame inválido")
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        self._last_frame_shape = (width, height)

        bytes_per_line = frame_rgb.strides[0]
        image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

        if text:
            self.info_label.setText(text)
        else:
            self.info_label.setText(f"{width}x{height}")

    def set_status(self, text: str) -> None:
        """Update status text below preview."""
        self.info_label.setText(text)

    def clear(self) -> None:
        """Clear current preview."""
        self.image_label.clear()
        self.image_label.setText("Sem imagem")
        self.info_label.setText("Pronto")

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        """Keep pixmap scaled when widget is resized."""
        super().resizeEvent(event)
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
