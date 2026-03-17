"""Inference engine entrypoint for image, video, and camera sources."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QObject, QTimer

from src.core.cascade_pipeline import CascadePipeline, CascadePipelineError
from src.core.model_loader import ModelLoader, ModelLoaderError
from src.inputs.camera_source import CameraSource, CameraSourceError
from src.inputs.image_source import ImageSource, ImageSourceError
from src.inputs.video_source import VideoSource, VideoSourceError
from src.ui.main_window import MainWindow


class InferenceEngine(QObject):
    """Coordinate source readers and model pipeline execution."""

    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self.window = window

        self.model_loader = ModelLoader()
        self.pipeline = CascadePipeline(self.model_loader)
        self.image_source = ImageSource()
        self.video_source = VideoSource()
        self.camera_source = CameraSource(device_index=0)

        self.source_type = "image"
        self.source_path = ""
        self.execution_mode = "cascade"
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        self.selected_model_ids: list[str] = []
        self.ordered_model_ids: list[str] = []
        self.running = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._process_stream_frame)

    def connect_ui(self) -> None:
        """Connect UI signals to engine actions."""
        self.window.modelLoadRequested.connect(self.load_model)
        self.window.modelRemoveRequested.connect(self.remove_model)
        self.window.modelOrderChanged.connect(self.set_model_order)
        self.window.modelSelectionChanged.connect(self.set_selected_models)
        self.window.sourceChanged.connect(self.set_source_type)
        self.window.sourcePathSelected.connect(self.set_source_path)
        self.window.executionModeChanged.connect(self.set_execution_mode)
        self.window.confidenceChanged.connect(self.set_confidence_threshold)
        self.window.startRequested.connect(self.start)
        self.window.stopRequested.connect(self.stop)

        self.window.set_models([], [])
        self.window.set_model_details({})
        self.window.set_execution_mode(self.execution_mode)
        self.window.set_legend("Legenda: sem classes detectadas.")
        self.window.set_status("Pronto. Carregue modelos e selecione uma fonte.")

    def load_model(self, model_path: str) -> None:
        """Load one model and refresh selector state."""
        try:
            path = Path(model_path)
            model_id = self._next_model_id(path.stem)
            loaded = self.model_loader.load(path, model_id=model_id)
        except ModelLoaderError as exc:
            self.window.set_status(f"Erro ao carregar modelo: {exc}")
            return

        self._refresh_model_state(default_select=loaded.model_id)
        self.window.set_status(f"Modelo carregado: {loaded.model_id} ({loaded.format})")

    def remove_model(self, model_id: str) -> None:
        """Unload model and update UI list."""
        try:
            self.model_loader.unload(model_id)
        except ModelLoaderError as exc:
            self.window.set_status(f"Erro ao remover modelo: {exc}")
            return

        self._refresh_model_state()
        self.window.set_status(f"Modelo removido: {model_id}")

    def set_model_order(self, model_ids: list[str]) -> None:
        """Set visual execution order from UI."""
        self.ordered_model_ids = list(model_ids)

    def set_selected_models(self, model_ids: list[str]) -> None:
        """Set checked/active models from UI."""
        self.selected_model_ids = list(model_ids)

    def set_source_type(self, source_type: str) -> None:
        """Set source type: image, video or camera."""
        self.source_type = source_type
        if source_type == "camera" and not self.source_path:
            self.source_path = "camera:0"

    def set_source_path(self, source_path: str) -> None:
        """Set selected source path."""
        self.source_path = source_path

    def set_execution_mode(self, mode: str) -> None:
        """Set model execution mode."""
        if mode not in {"cascade", "simultaneo"}:
            self.window.set_status(f"Modo de execução inválido: {mode}")
            return
        self.execution_mode = mode

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold used during prediction."""
        self.confidence_threshold = max(0.01, min(1.0, float(threshold)))
        self.pipeline.confidence_threshold = self.confidence_threshold

    def start(self) -> None:
        """Start inference according to selected source."""
        if self.running:
            return

        active_order = self._active_model_order()
        if not active_order:
            self.window.set_status("Selecione ao menos um modelo.")
            self.window.set_running(False)
            return

        self.pipeline.set_model_order(active_order)
        self.pipeline.confidence_threshold = self.confidence_threshold
        self.pipeline.iou_threshold = self.iou_threshold

        if self.source_type == "image":
            self._process_image_once()
            self.window.set_running(False)
            return

        if self.source_type == "video":
            self._start_video_stream()
            return

        if self.source_type == "camera":
            self._start_camera_stream()
            return

        self.window.set_status(f"Tipo de fonte inválido: {self.source_type}")
        self.window.set_running(False)

    def stop(self) -> None:
        """Stop active stream and release resources."""
        self._timer.stop()
        self.video_source.release()
        self.camera_source.stop()
        self.running = False
        self.window.set_running(False)

    def _process_image_once(self) -> None:
        if not self.source_path:
            self.window.set_status("Selecione uma imagem antes de iniciar.")
            return
        try:
            frame = self.image_source.read(self.source_path)
            annotated, total = self._infer_and_annotate(frame)
            self.window.show_frame(annotated, f"Imagem processada - detecções: {total}")
        except (ImageSourceError, CascadePipelineError, ModelLoaderError) as exc:
            self.window.set_status(f"Erro na inferência de imagem: {exc}")

    def _start_video_stream(self) -> None:
        if not self.source_path:
            self.window.set_status("Selecione um vídeo antes de iniciar.")
            self.window.set_running(False)
            return
        try:
            self.video_source.open(self.source_path)
            metadata = self.video_source.metadata()
        except VideoSourceError as exc:
            self.window.set_status(f"Erro ao abrir vídeo: {exc}")
            self.window.set_running(False)
            return

        fps = metadata.get("fps", 0.0) or 30.0
        interval_ms = max(1, int(1000 / fps))
        self.running = True
        self._timer.start(interval_ms)
        self.window.set_running(True)
        self.window.set_status(f"Vídeo iniciado: {metadata.get('path', '')}")

    def _start_camera_stream(self) -> None:
        index = 0
        if self.source_path.startswith("camera:"):
            try:
                index = int(self.source_path.split(":", maxsplit=1)[1])
            except ValueError:
                index = 0
        self.camera_source.device_index = index

        try:
            self.camera_source.start()
        except CameraSourceError as exc:
            self.window.set_status(f"Erro ao abrir câmera: {exc}")
            self.window.set_running(False)
            return

        self.running = True
        self._timer.start(33)
        self.window.set_running(True)
        self.window.set_status(f"Câmera iniciada (dispositivo {index}).")

    def _process_stream_frame(self) -> None:
        try:
            if self.source_type == "video":
                success, frame = self.video_source.read()
            elif self.source_type == "camera":
                success, frame = self.camera_source.read()
            else:
                self.stop()
                return

            if not success or frame is None:
                self.stop()
                self.window.set_status("Fluxo finalizado.")
                return

            annotated, total = self._infer_and_annotate(frame)
            self.window.show_frame(annotated, f"Detecções: {total}")
        except (VideoSourceError, CameraSourceError, CascadePipelineError, ModelLoaderError) as exc:
            self.stop()
            self.window.set_status(f"Erro durante inferência: {exc}")

    def _infer_and_annotate(self, frame_bgr: Any) -> tuple[Any, int]:
        detections: list[dict[str, Any]]
        if self.execution_mode == "cascade":
            result = self.pipeline.run(frame_bgr)
            detections = result.get("all_detections", [])
        else:
            detections = self._run_models_simultaneously(frame_bgr)

        frame = frame_bgr.copy()
        legend_map: dict[str, tuple[tuple[int, int, int], str]] = {}

        for detection in detections:
            bbox = detection.get("bbox", [])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            label = str(detection.get("label", "obj"))
            confidence = float(detection.get("confidence", 0.0))
            class_id = int(detection.get("class_id", -1))
            model_id = str(detection.get("model_id", "model"))
            color = self._class_color(model_id, class_id, label)
            legend_key = f"{model_id}:{label}:{class_id}"
            legend_map[legend_key] = (color, f"{model_id} :: {label}")

            # Improved readability: thicker lines, anti-aliasing, and colored label background.
            thickness = max(2, int(min(frame.shape[0], frame.shape[1]) / 320))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            _ = confidence  # confidence remains available for future visual tuning.

        self.window.set_legend(self._build_legend_text(legend_map))
        return frame, len(detections)

    def _run_models_simultaneously(self, frame_bgr: Any) -> list[dict[str, Any]]:
        """Run all selected models on the same frame (non-cascade mode)."""
        detections: list[dict[str, Any]] = []
        next_id = 1
        for model_id in self._active_model_order():
            model = self.model_loader.get(model_id)
            raw = model.predict(
                frame_bgr,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
            )
            normalized = self.pipeline._normalize_detections(raw)
            for item in normalized:
                entry = {
                    "id": next_id,
                    "stage_index": 0,
                    "model_id": model_id,
                    "parent_detection_id": None,
                    "bbox": item.get("bbox", []),
                    "confidence": item.get("confidence", 0.0),
                    "class_id": item.get("class_id", -1),
                    "label": item.get("label", "obj"),
                }
                detections.append(entry)
                next_id += 1
        return detections

    @staticmethod
    def _class_color(model_id: str, class_id: int, label: str) -> tuple[int, int, int]:
        """Return deterministic BGR color per class, including model namespace."""
        key = f"{model_id}:{class_id}:{label}".encode("utf-8", errors="ignore")
        digest = hashlib.md5(key).digest()
        b = int(digest[0])
        g = int(digest[1])
        r = int(digest[2])
        # Keep colors vivid and avoid too-dark tones.
        return (max(40, b), max(40, g), max(40, r))

    @staticmethod
    def _build_legend_text(legend_map: dict[str, tuple[tuple[int, int, int], str]]) -> str:
        """Build text legend mapping class and model to the box color."""
        if not legend_map:
            return "Legenda: sem classes detectadas."

        lines = ["<b>Legenda (classe - cor):</b>"]
        for _, (color, title) in sorted(legend_map.items(), key=lambda item: item[1][1].lower()):
            b, g, r = color
            lines.append(
                (
                    "<div>"
                    f"<span style='color: rgb({r},{g},{b}); font-weight: bold;'>&#9632;</span> "
                    f"{title} "
                    f"<span style='color:#aaa;'>(RGB {r},{g},{b})</span>"
                    "</div>"
                )
            )
        return "".join(lines)

    def _next_model_id(self, base: str) -> str:
        existing = {entry["model_id"] for entry in self.model_loader.list_models()}
        if base not in existing:
            return base
        counter = 2
        while f"{base}_{counter}" in existing:
            counter += 1
        return f"{base}_{counter}"

    def _refresh_model_state(self, default_select: str | None = None) -> None:
        loaded_ids = [entry["model_id"] for entry in self.model_loader.list_models()]
        kept_order = [model_id for model_id in self.ordered_model_ids if model_id in loaded_ids]
        for model_id in loaded_ids:
            if model_id not in kept_order:
                kept_order.append(model_id)
        self.ordered_model_ids = kept_order

        kept_selected = [model_id for model_id in self.selected_model_ids if model_id in loaded_ids]
        if default_select and default_select in loaded_ids and default_select not in kept_selected:
            kept_selected.append(default_select)
        self.selected_model_ids = kept_selected

        self.window.set_models(self.ordered_model_ids, self.selected_model_ids)
        self.window.set_model_details(self._build_model_details())

    def _active_model_order(self) -> list[str]:
        selected = set(self.selected_model_ids)
        return [model_id for model_id in self.ordered_model_ids if model_id in selected]

    def _build_model_details(self) -> dict[str, str]:
        details: dict[str, str] = {}
        for entry in self.model_loader.list_models():
            model_id = str(entry.get("model_id", ""))
            model_path = str(entry.get("path", ""))
            model_format = str(entry.get("format", ""))
            model_backend = str(entry.get("backend", ""))
            classes = entry.get("classes", [])

            if isinstance(classes, list) and classes:
                class_lines = "\n".join(f"  - {name}" for name in classes[:120])
            else:
                class_lines = "  - Classes não disponíveis para este backend/modelo."

            details[model_id] = (
                f"ID: {model_id}\n"
                f"Formato: {model_format}\n"
                f"Backend: {model_backend}\n"
                f"Caminho: {model_path}\n"
                f"Classes ({len(classes) if isinstance(classes, list) else 0}):\n"
                f"{class_lines}"
            )
        return details
