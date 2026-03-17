"""Cascade pipeline orchestration for multi-model detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.model_loader import ModelLoader


class CascadePipelineError(Exception):
    """Raised for invalid cascade configuration or runtime failures."""


@dataclass(slots=True)
class _CascadeInput:
    """Internal input unit for each stage."""

    image: Any
    offset_x: float
    offset_y: float
    parent_detection_id: int | None = None


class CascadePipeline:
    """Run one or more models in cascade mode."""

    def __init__(
        self,
        model_loader: ModelLoader,
        model_order: list[str] | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.model_loader = model_loader
        self.model_order = model_order or []
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def set_model_order(self, model_order: list[str]) -> None:
        """Set model execution order used by cascade."""
        self.model_order = model_order

    def run(self, image: Any, model_order: list[str] | None = None) -> dict[str, Any]:
        """Execute cascade flow and return detections per stage.

        The first model runs on the full image. For each next stage, detections
        from the previous stage are cropped and used as inputs.
        """
        if image is None:
            raise CascadePipelineError("Image input cannot be None.")

        order = model_order if model_order is not None else self.model_order
        if not order:
            raise CascadePipelineError("No model order configured for cascade.")

        current_inputs = [_CascadeInput(image=image, offset_x=0.0, offset_y=0.0)]
        all_detections: list[dict[str, Any]] = []
        stages: list[dict[str, Any]] = []
        next_detection_id = 1

        for stage_index, model_id in enumerate(order):
            if not current_inputs:
                stages.append(
                    {
                        "stage_index": stage_index,
                        "model_id": model_id,
                        "inputs_count": 0,
                        "detections": [],
                        "stopped_early": True,
                    }
                )
                break

            model = self.model_loader.get(model_id)
            stage_detections: list[dict[str, Any]] = []
            next_inputs: list[_CascadeInput] = []

            for stage_input in current_inputs:
                raw_predictions = model.predict(
                    stage_input.image,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                )
                normalized = self._normalize_detections(raw_predictions)

                for detection in normalized:
                    local_bbox = self._clip_bbox(
                        detection["bbox"],
                        image_shape=stage_input.image.shape,
                    )
                    if local_bbox is None:
                        continue

                    x1_local, y1_local, x2_local, y2_local = local_bbox
                    x1_global = x1_local + stage_input.offset_x
                    y1_global = y1_local + stage_input.offset_y
                    x2_global = x2_local + stage_input.offset_x
                    y2_global = y2_local + stage_input.offset_y

                    detection_item = {
                        "id": next_detection_id,
                        "stage_index": stage_index,
                        "model_id": model_id,
                        "parent_detection_id": stage_input.parent_detection_id,
                        "bbox": [x1_global, y1_global, x2_global, y2_global],
                        "bbox_local": [x1_local, y1_local, x2_local, y2_local],
                        "confidence": detection["confidence"],
                        "class_id": detection["class_id"],
                        "label": detection["label"],
                    }
                    next_detection_id += 1
                    stage_detections.append(detection_item)
                    all_detections.append(detection_item)

                    crop = stage_input.image[int(y1_local) : int(y2_local), int(x1_local) : int(x2_local)]
                    if crop.size == 0:
                        continue
                    next_inputs.append(
                        _CascadeInput(
                            image=crop,
                            offset_x=x1_global,
                            offset_y=y1_global,
                            parent_detection_id=detection_item["id"],
                        )
                    )

            stages.append(
                {
                    "stage_index": stage_index,
                    "model_id": model_id,
                    "inputs_count": len(current_inputs),
                    "detections": stage_detections,
                    "stopped_early": False,
                }
            )
            current_inputs = next_inputs

        return {
            "mode": "cascade",
            "model_order": order,
            "stages": stages,
            "all_detections": all_detections,
        }

    def _normalize_detections(self, raw_predictions: Any) -> list[dict[str, Any]]:
        """Convert backend-specific output to a common list format."""
        if raw_predictions is None:
            return []

        # Ultralytics usually returns a list of Results objects.
        if isinstance(raw_predictions, list) and raw_predictions and hasattr(raw_predictions[0], "boxes"):
            return self._normalize_ultralytics(raw_predictions)

        # Generic list[dict] fallback.
        if isinstance(raw_predictions, list) and all(isinstance(item, dict) for item in raw_predictions):
            return self._normalize_dict_detections(raw_predictions)

        # ONNX/generic tensor-like output fallback.
        if isinstance(raw_predictions, list) and raw_predictions:
            first = raw_predictions[0]
            if hasattr(first, "shape"):
                return self._normalize_array(first)

        if hasattr(raw_predictions, "shape"):
            return self._normalize_array(raw_predictions)

        return []

    def _normalize_ultralytics(self, results: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or getattr(boxes, "xyxy", None) is None:
                continue

            names = getattr(result, "names", {})
            xyxy = self._to_python_array(boxes.xyxy)
            confs = self._to_python_array(getattr(boxes, "conf", []))
            classes = self._to_python_array(getattr(boxes, "cls", []))

            for index, bbox in enumerate(xyxy):
                if len(bbox) < 4:
                    continue
                confidence = float(confs[index]) if index < len(confs) else 0.0
                if confidence < self.confidence_threshold:
                    continue
                class_id = int(classes[index]) if index < len(classes) else -1
                label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                normalized.append(
                    {
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "confidence": confidence,
                        "class_id": class_id,
                        "label": label,
                    }
                )
        return normalized

    def _normalize_dict_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in detections:
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            confidence = float(item.get("confidence", 0.0))
            if confidence < self.confidence_threshold:
                continue
            class_id = int(item.get("class_id", -1))
            label = str(item.get("label", class_id))
            normalized.append(
                {
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "confidence": confidence,
                    "class_id": class_id,
                    "label": label,
                }
            )
        return normalized

    def _normalize_array(self, array_like: Any) -> list[dict[str, Any]]:
        rows = self._to_python_array(array_like)
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 4:
                continue
            confidence = float(row[4]) if len(row) > 4 else 1.0
            if confidence < self.confidence_threshold:
                continue
            class_id = int(row[5]) if len(row) > 5 else -1
            normalized.append(
                {
                    "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                    "confidence": confidence,
                    "class_id": class_id,
                    "label": str(class_id),
                }
            )
        return normalized

    @staticmethod
    def _to_python_array(value: Any) -> list[Any]:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _clip_bbox(bbox: list[float], image_shape: Any) -> tuple[float, float, float, float] | None:
        if len(bbox) < 4 or image_shape is None or len(image_shape) < 2:
            return None

        height, width = int(image_shape[0]), int(image_shape[1])
        x1, y1, x2, y2 = bbox[:4]

        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))

        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2
