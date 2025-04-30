import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from .preprocessing import get_clip_transform, validate_preprocessing


class ArtStyleDataset(Dataset):
    """Dataset for art style text-image pairs"""

    def __init__(self, csv_path, image_root, transform=None):
        validate_preprocessing(transform)
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self._preprocess_text()

    def _preprocess_text(self):
        """Convert style labels to descriptive text"""
        self.df['text'] = self.df['style'].apply(
            lambda x: f"A painting in the {x} style, professional artwork, trending on artstation"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = f"{self.image_root}/{row['style']}/{row['image_filename']}"

        try:
            image = Image.open(image_path).convert("RGB")
            text = row['text']
        except Exception as e:
            print(f"Error loading {image_path}: {str(e)}")
            return self[(idx + 1) % len(self)]  # Skip corrupted files

        if self.transform:
            image = self.transform(image)

        return image, text