from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from PIL import Image

from scene_pipeline.schemas import FrameMetadata, LabelScore


class TorchSceneClassifier:
    """TorchVision image classifier with scene-level probability aggregation."""

    def __init__(
        self,
        model_name: str = "resnet50",
        device: str = "auto",
        top_k: int = 5,
        batch_size: int = 8,
        pretrained: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device_name = device
        self.top_k = top_k
        self.batch_size = batch_size
        self.pretrained = pretrained
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        try:
            import torch
            from torchvision.models import (
                EfficientNet_B0_Weights,
                ResNet50_Weights,
                efficientnet_b0,
                resnet50,
            )
            from torchvision.transforms import Compose, Normalize, Resize, ToTensor
        except ImportError as exc:
            raise RuntimeError("PyTorch and TorchVision are required for classification") from exc

        self.torch = torch
        if self.device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_name)

        normalized_name = self.model_name.lower().replace("-", "_")
        if normalized_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if self.pretrained else None
            self.model = efficientnet_b0(weights=weights)
        elif normalized_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if self.pretrained else None
            self.model = resnet50(weights=weights)
        else:
            raise ValueError("model_name must be 'resnet50' or 'efficientnet_b0'")

        if weights is not None:
            self.transform = weights.transforms()
            self.categories = list(weights.meta["categories"])
        else:
            self.transform = Compose(
                [
                    Resize((224, 224)),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            self.categories = [f"class_{index}" for index in range(1000)]

        self.model.eval()
        self.model.to(self.device)
        self._loaded = True

    def classify_scene(self, frames: list[FrameMetadata]) -> list[LabelScore]:
        if not frames:
            return []

        self._load()
        probabilities_by_label: dict[str, list[float]] = defaultdict(list)

        for start in range(0, len(frames), self.batch_size):
            batch_frames = frames[start : start + self.batch_size]
            images = []
            for frame in batch_frames:
                image = Image.open(Path(frame.path)).convert("RGB")
                images.append(self.transform(image))
            batch = self.torch.stack(images).to(self.device)
            with self.torch.no_grad():
                logits = self.model(batch)
                probabilities = self.torch.nn.functional.softmax(logits, dim=1)
                values, indices = probabilities.topk(self.top_k, dim=1)

            for row_values, row_indices in zip(values.cpu().tolist(), indices.cpu().tolist()):
                for confidence, label_index in zip(row_values, row_indices):
                    label = self.categories[label_index]
                    probabilities_by_label[label].append(float(confidence))

        aggregated = [
            LabelScore(label=label, confidence=sum(scores) / len(scores))
            for label, scores in probabilities_by_label.items()
        ]
        return sorted(aggregated, key=lambda item: item.confidence, reverse=True)[: self.top_k]


class NullSceneClassifier:
    """Development fallback that keeps the pipeline testable without model packages."""

    def classify_scene(self, frames: list[FrameMetadata]) -> list[LabelScore]:
        confidence = 1.0 if frames else 0.0
        return [LabelScore(label="unclassified", confidence=confidence)]


def build_classifier(
    model_name: str,
    device: str,
    top_k: int,
    batch_size: int,
) -> TorchSceneClassifier:
    return TorchSceneClassifier(
        model_name=model_name,
        device=device,
        top_k=top_k,
        batch_size=batch_size,
    )

