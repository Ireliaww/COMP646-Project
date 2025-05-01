import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from PIL import Image, ImageOps
import os


def visualize_dataset(csv_path, image_root, save_path="dataset_samples.png", target_size=(256, 256)):
    """
    Visualize random 9 samples from the dataset in a 3x3 grid
    可视化数据集中的随机9个样本(3x3网格)

    Args:
        csv_path (str): Path to annotations CSV file
        image_root (str): Root directory of images
        save_path (str): Path to save output figure
    """
    # Load annotations
    df = pd.read_csv(csv_path)

    # Randomly select 9 samples
    selected_samples = df.sample(n=9, random_state=99)  # Fixed seed for reproducibility

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for idx, (ax, (_, row)) in enumerate(zip(axes, selected_samples.iterrows())):
        try:
            # Load image
            img_dir = os.path.join(image_root, row['style'])
            img_path = os.path.join(img_dir, row['image_filename'])
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS,
                               bleed=0.2, centering=(0.5, 0.5))
            # Preprocessing inversion (if needed)
            img = np.array(img)  # Convert to numpy array

            # For visualization purpose, we assume:
            # - Images are stored in 0-255 range
            # - No normalization was applied
            # If you applied preprocessing normalization, add inverse normalization here

            # Show image
            ax.imshow(img)
            ax.set_title(f"Style: {row['style']}", fontsize=10)
            ax.axis('off')

        except Exception as e:
            print(f"Error loading {row['image_filename']}: {str(e)}")
            ax.axis('off')
            ax.text(0.5, 0.5, 'Image load failed',
                    horizontalalignment='center',
                    verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.show()


# 使用示例
if __name__ == "__main__":
    csv_path = "../../Datasets/Wiki/annotations.csv"
    image_root = "../../Datasets/Wiki"

    visualize_dataset(csv_path, image_root, save_path="dataset_samples.png", target_size=(256, 256))