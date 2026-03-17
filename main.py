"""Application entrypoint for predict_system."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from src.core.inference_engine import InferenceEngine
from src.ui.main_window import MainWindow


def main() -> int:
    """Start the desktop application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    engine = InferenceEngine(window)
    engine.connect_ui()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
