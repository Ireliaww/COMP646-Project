from typing import Dict, Tuple, Any, List
import torch
import clip
from torch import nn


class CLIPMetrics:
    """Metric calculation module for CLIP evaluation"""

    def __init__(self, device="cuda", clip_model="ViT-B/32"):
        self.device = device
        self.model, _ = clip.load(clip_model, device=device)
        self.model.eval()
        self.null_image_features = None
        self.null_text_features = None

    def _precompute_null_features(self, batch_size: int):
        """Precompute null features for CLIP score baseline"""
        with torch.no_grad():
            null_text = clip.tokenize([""] * batch_size).to(self.device)
            null_image = torch.zeros(batch_size, 3, 224, 224).to(self.device)
            self.null_text_features = self.model.encode_text(null_text)
            self.null_image_features = self.model.encode_image(null_image)

    def clip_score(self, images: torch.Tensor, texts: List[str]) -> Tuple[float, float]:
        """Calculate directional CLIP scores"""
        batch_size = images.size(0)
        if self.null_image_features is None or self.null_text_features.size(0) != batch_size:
            self._precompute_null_features(batch_size)

        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(self.device)
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text_tokens)

        image_sim = (image_features - self.null_image_features) @ (text_features - self.null_text_features).T
        text_sim = (text_features - self.null_text_features) @ (image_features - self.null_image_features).T

        return image_sim.diag().mean().item(), text_sim.diag().mean().item()