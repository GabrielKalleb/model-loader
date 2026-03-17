"""Model loading abstractions for .pt, .onnx, and other formats."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class ModelLoaderError(Exception):
    """Raised when loading or accessing models fails."""


@dataclass(slots=True)
class LoadedModel:
    """Represents one loaded model with a unified predict interface."""

    model_id: str
    path: Path
    format: str
    backend: str
    model: Any
    class_names: list[str]
    _predict_fn: Callable[..., Any] = field(repr=False)

    def predict(self, input_data: Any, **kwargs: Any) -> Any:
        """Run inference using the loaded model."""
        return self._predict_fn(input_data, **kwargs)


class ModelLoader:
    """Load and register inference models."""

    SUPPORTED_FORMATS = {".pt", ".onnx"}

    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}

    def load(self, model_path: str | Path, model_id: str | None = None, backend: str = "auto") -> LoadedModel:
        """Load a model from disk and register it.

        Args:
            model_path: Path to model file (.pt or .onnx).
            model_id: Optional custom id. If omitted, file stem is used.
            backend: Preferred backend. For .pt, options are:
                - "auto" (default): tries ultralytics then torch.
                - "ultralytics": force YOLO backend.
                - "torch": force generic torch backend.
                For .onnx, backend is always ONNX Runtime.
        """
        path = Path(model_path)
        if not path.exists():
            raise ModelLoaderError(f"Model file not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ModelLoaderError(
                f"Unsupported model format '{suffix}'. Supported: {sorted(self.SUPPORTED_FORMATS)}"
            )

        resolved_id = model_id or path.stem
        if resolved_id in self._models:
            raise ModelLoaderError(f"Model id already loaded: {resolved_id}")

        if suffix == ".pt":
            loaded = self._load_pt(path, resolved_id, backend=backend)
        elif suffix == ".onnx":
            loaded = self._load_onnx(path, resolved_id)
        else:
            raise ModelLoaderError(f"No loader implemented for: {suffix}")

        self._models[resolved_id] = loaded
        return loaded

    def unload(self, model_id: str) -> None:
        """Remove model from registry."""
        if model_id not in self._models:
            raise ModelLoaderError(f"Model id not found: {model_id}")
        del self._models[model_id]

    def get(self, model_id: str) -> LoadedModel:
        """Get loaded model by id."""
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise ModelLoaderError(f"Model id not found: {model_id}") from exc

    def list_models(self) -> list[dict[str, Any]]:
        """Return metadata for all loaded models."""
        return [
            {
                "model_id": model.model_id,
                "path": str(model.path),
                "format": model.format,
                "backend": model.backend,
                "classes": model.class_names,
            }
            for model in self._models.values()
        ]

    def _load_pt(self, path: Path, model_id: str, backend: str = "auto") -> LoadedModel:
        allowed = {"auto", "ultralytics", "torch"}
        if backend not in allowed:
            raise ModelLoaderError(f"Invalid backend '{backend}' for .pt. Allowed: {sorted(allowed)}")

        if backend in {"auto", "ultralytics"}:
            try:
                from ultralytics import YOLO  # type: ignore

                model = YOLO(str(path))
                raw_names = getattr(model, "names", {})
                if isinstance(raw_names, dict):
                    class_names = [str(raw_names[key]) for key in sorted(raw_names)]
                elif isinstance(raw_names, list):
                    class_names = [str(name) for name in raw_names]
                else:
                    class_names = []

                def predict_fn(input_data: Any, **kwargs: Any) -> Any:
                    conf = kwargs.get("conf", 0.25)
                    iou = kwargs.get("iou", 0.45)
                    device = kwargs.get("device", None)
                    return model.predict(source=input_data, conf=conf, iou=iou, device=device, verbose=False)

                return LoadedModel(
                    model_id=model_id,
                    path=path,
                    format=".pt",
                    backend="ultralytics",
                    model=model,
                    class_names=class_names,
                    _predict_fn=predict_fn,
                )
            except Exception as ul_err:
                if backend == "ultralytics":
                    raise ModelLoaderError(f"Failed to load .pt with ultralytics: {ul_err}") from ul_err

        try:
            import torch  # type: ignore

            try:
                model = torch.jit.load(str(path), map_location="cpu")
            except Exception:
                model = torch.load(str(path), map_location="cpu")

            if hasattr(model, "eval"):
                model.eval()

            def predict_fn(input_data: Any, **kwargs: Any) -> Any:
                with torch.inference_mode():
                    return model(input_data, **kwargs) if callable(model) else model

            return LoadedModel(
                model_id=model_id,
                path=path,
                format=".pt",
                backend="torch",
                model=model,
                class_names=[],
                _predict_fn=predict_fn,
            )
        except Exception as torch_err:
            raise ModelLoaderError(f"Failed to load .pt with torch fallback: {torch_err}") from torch_err

    def _load_onnx(self, path: Path, model_id: str) -> LoadedModel:
        try:
            import onnxruntime as ort  # type: ignore

            available = ort.get_available_providers()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            selected = [provider for provider in providers if provider in available]
            if not selected:
                selected = available

            session = ort.InferenceSession(str(path), providers=selected)
            input_names = [item.name for item in session.get_inputs()]

            def predict_fn(input_data: Any, **kwargs: Any) -> Any:
                if isinstance(input_data, dict):
                    feeds = input_data
                else:
                    if not input_names:
                        raise ModelLoaderError("ONNX model has no input nodes.")
                    feeds = {input_names[0]: input_data}
                output_names = kwargs.get("output_names", None)
                return session.run(output_names, feeds)

            return LoadedModel(
                model_id=model_id,
                path=path,
                format=".onnx",
                backend="onnxruntime",
                model=session,
                class_names=[],
                _predict_fn=predict_fn,
            )
        except Exception as exc:
            raise ModelLoaderError(f"Failed to load ONNX model: {exc}") from exc
