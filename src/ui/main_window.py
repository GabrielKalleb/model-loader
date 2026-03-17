"""Main desktop window."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.ui.model_selector import ModelSelector
from src.ui.preview_panel import PreviewPanel


class MainWindow(QMainWindow):
    """Root UI container."""

    modelLoadRequested = Signal(str)
    modelRemoveRequested = Signal(str)
    modelOrderChanged = Signal(list)
    modelSelectionChanged = Signal(list)
    sourceChanged = Signal(str)
    sourcePathSelected = Signal(str)
    executionModeChanged = Signal(str)
    confidenceChanged = Signal(float)
    startRequested = Signal()
    stopRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("predict_system - Detecção")
        self.resize(1200, 760)
        self._build_ui()
        self._wire_events()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QSplitter()
        root.addWidget(splitter, stretch=1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.model_selector = ModelSelector()
        left_layout.addWidget(self.model_selector, stretch=1)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_panel = PreviewPanel()
        right_layout.addWidget(self.preview_panel, stretch=1)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("Fonte:"))

        self.source_combo = QComboBox()
        self.source_combo.addItems(["image", "video", "camera"])
        control_row.addWidget(self.source_combo)

        self.open_source_button = QPushButton("Selecionar arquivo")
        control_row.addWidget(self.open_source_button)

        control_row.addWidget(QLabel("Execução:"))
        self.execution_mode_combo = QComboBox()
        self.execution_mode_combo.addItems(["cascade", "simultaneo"])
        self.execution_mode_combo.setToolTip("cascade: respeita ordem; simultaneo: todos os modelos no frame inteiro")
        control_row.addWidget(self.execution_mode_combo)

        control_row.addWidget(QLabel("Confiança:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.01, 1.00)
        self.confidence_spin.setSingleStep(0.01)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(0.25)
        control_row.addWidget(self.confidence_spin)

        self.start_button = QPushButton("Iniciar")
        self.stop_button = QPushButton("Parar")
        self.stop_button.setEnabled(False)
        control_row.addWidget(self.start_button)
        control_row.addWidget(self.stop_button)

        right_layout.addLayout(control_row)
        splitter.addWidget(right)
        splitter.setSizes([320, 880])

        self.legend_label = QLabel("Legenda: sem classes detectadas.")
        self.legend_label.setTextFormat(Qt.RichText)
        self.legend_label.setWordWrap(True)
        self.legend_label.setStyleSheet("padding: 4px; border: 1px solid #444;")
        root.addWidget(self.legend_label)
        self.status_label = QLabel("Pronto para carregar modelos.")
        root.addWidget(self.status_label)

        self.setCentralWidget(central)

    def _wire_events(self) -> None:
        self.model_selector.orderChanged.connect(self.modelOrderChanged.emit)
        self.model_selector.selectionChanged.connect(self.modelSelectionChanged.emit)
        self.model_selector.loadClicked.connect(self._request_model_file)
        self.model_selector.removeClicked.connect(self.modelRemoveRequested.emit)

        self.source_combo.currentTextChanged.connect(self.sourceChanged.emit)
        self.open_source_button.clicked.connect(self._select_source_file)
        self.execution_mode_combo.currentTextChanged.connect(self._on_execution_mode_changed)
        self.confidence_spin.valueChanged.connect(self.confidenceChanged.emit)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)

    def set_models(self, all_model_ids: list[str], selected_model_ids: list[str] | None = None) -> None:
        """Refresh model selector contents."""
        self.model_selector.set_models(all_model_ids, selected_model_ids)

    def set_model_details(self, details: dict[str, str]) -> None:
        """Set description text for each loaded model."""
        self.model_selector.set_model_details(details)

    def set_status(self, text: str) -> None:
        """Update status text in footer."""
        self.status_label.setText(text)
        self.preview_panel.set_status(text)

    def set_legend(self, text: str) -> None:
        """Set legend content with class-color mapping."""
        self.legend_label.setText(text)

    def show_frame(self, frame_bgr: Any, info: str | None = None) -> None:
        """Display frame in preview panel."""
        self.preview_panel.update_frame(frame_bgr, text=info)

    def set_running(self, running: bool) -> None:
        """Reflect engine running state in controls."""
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def set_execution_mode(self, mode: str) -> None:
        """Set execution mode and update selector behavior."""
        index = self.execution_mode_combo.findText(mode)
        if index >= 0 and self.execution_mode_combo.currentIndex() != index:
            self.execution_mode_combo.setCurrentIndex(index)
        self.model_selector.set_ordering_enabled(mode == "cascade")

    def _request_model_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar modelo",
            "",
            "Modelos (*.pt *.onnx);;Todos os arquivos (*)",
        )
        if not file_path:
            return
        self.modelLoadRequested.emit(file_path)

    def _select_source_file(self) -> None:
        source_type = self.source_combo.currentText()
        if source_type == "image":
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Selecionar imagem",
                "",
                "Imagens (*.jpg *.jpeg *.png *.bmp *.webp);;Todos os arquivos (*)",
            )
        elif source_type == "video":
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Selecionar vídeo",
                "",
                "Vídeos (*.mp4 *.avi *.mov *.mkv);;Todos os arquivos (*)",
            )
        else:
            file_path = "camera:0"
        if file_path:
            self.sourcePathSelected.emit(file_path)
            self.set_status(f"Fonte selecionada: {file_path}")

    def _on_start_clicked(self) -> None:
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.startRequested.emit()
        self.set_status("Inferência iniciada.")

    def _on_stop_clicked(self) -> None:
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.stopRequested.emit()
        self.set_status("Inferência parada.")

    def _on_execution_mode_changed(self, mode: str) -> None:
        self.model_selector.set_ordering_enabled(mode == "cascade")
        self.executionModeChanged.emit(mode)
