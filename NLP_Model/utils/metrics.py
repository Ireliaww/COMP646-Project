import numpy as np
import torch
import clip
from typing import List, Dict

class CLIPMetrics:
    def __init__(self, device: str = "cuda", clip_model: str = "ViT-B/32"):
        """CLIP-based metric calculator for text-image alignment
        Args:
            device: Computation device (cuda/cpu)
            clip_model: CLIP model variant
        """
        self.device = device
        self.model, _ = clip.load(clip_model, device=device)
        self.model.eval()
        self.model.requires_grad_(False)  # Freeze all parameters

    def compute_all_metrics(self,
                            images: torch.Tensor,
                            texts: List[str]) -> Dict[str, float]:
        """Calculate key alignment metrics between images and texts

        Args:
            images: Preprocessed image tensor [N, C, H, W]
            texts: Raw text descriptions (list of strings)

        Returns:
            Dictionary containing:
            - clip_score: Average cosine similarity (0-1)
            - top1_acc: Retrieval accuracy (%)
        """
        # Validate input dimensions
        if images.dim() != 4:
            raise ValueError(f"Images must be 4D tensor, got {images.dim()}D")

        if len(texts) != images.size(0):
            raise ValueError(f"Mismatched samples: {len(texts)} texts vs {images.size(0)} images")

        # Tokenize texts and move to device
        text_inputs = clip.tokenize(texts).to(self.device)

        with torch.no_grad():
            # Extract normalized features
            image_features = self.model.encode_image(images.to(self.device))
            text_features = self.model.encode_text(text_inputs)

            # L2 normalization
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity matrix
            similarity = image_features @ text_features.T  # [N, N]

            # CLIP score (diagonal elements)
            clip_score = torch.diag(similarity).mean().item()

            # Top-1 accuracy
            labels = torch.arange(len(texts), device=self.device)
            predictions = similarity.argmax(dim=1)
            top1_acc = (predictions == labels).float().mean().item() * 100

        return {
            "clip_score": clip_score,
            "top1_acc": top1_acc
        }