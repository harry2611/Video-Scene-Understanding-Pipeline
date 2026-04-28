from __future__ import annotations

import math
from pathlib import Path

from PIL import Image


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class CLIPSceneEmbedder:
    """CLIP image/text embedding helper used for semantic scene search."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device_name = device
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for CLIP embeddings") from exc

        self.torch = torch
        if self.device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self._loaded = True

    def embed_image(self, image_path: str | Path) -> list[float]:
        self._load()
        image = Image.open(Path(image_path)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return [float(value) for value in features[0].cpu().tolist()]

    def embed_text(self, text: str) -> list[float]:
        self._load()
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return [float(value) for value in features[0].cpu().tolist()]

