# data/preprocessing.py

import torch
import clip
from PIL import Image
from torchvision import transforms


def get_clip_transform(train: bool = False) -> transforms.Compose:
    """
    CLIP-specific image preprocessing pipeline.

    Args:
        train: Whether to include augmentations for training.
              Uses standard preprocessing for validation/generation.

    Returns:
        A torchvision Compose object with preprocessing steps.
    """
    # Official CLIP preprocessing parameters
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )

    # Base preprocessing for all modes
    base_transform = [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]

    # Add augmentations only for training
    if train:
        base_transform = [
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.01)], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            *base_transform[2:]  # Keep ToTensor and Normalize
        ]

    return transforms.Compose(base_transform)


def clip_tokenize(text: str, context_length: int = 77) -> torch.Tensor:
    """
    Tokenize text input for CLIP model.

    Args:
        text: Input text string
        context_length: Maximum sequence length (CLIP default is 77)

    Returns:
        Tokenized tensor with shape [1, context_length]
    """
    return clip.tokenize([text], truncate=True, context_length=context_length)


def validate_preprocessing(image: Image.Image) -> bool:
    """
    Validate image meets preprocessing requirements.

    Args:
        image: PIL Image object

    Returns:
        True if valid, False if corrupted or invalid format
    """
    try:
        image.verify()
        if image.mode != "RGB":
            return False
        return True
    except Exception:
        return False