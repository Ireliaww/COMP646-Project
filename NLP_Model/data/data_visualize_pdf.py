import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import os

def visualize_dataset(
    csv_path: str,
    image_root: str,
    save_path: str = "dataset_samples.pdf",
    target_size: tuple[int, int] = (256, 256)
):
    """
    Visualize 9 random samples from the dataset in a 3x3 grid and
    export the figure as a vector PDF to ensure that text is selectable.

    Args:
        csv_path (str): Path to the annotations CSV file.
        image_root (str): Root directory containing image subfolders.
        save_path (str): Output file path for the figure (should end with .pdf).
        target_size (tuple): Target size (width, height) to resize each image.
    """
    # Load annotation data from CSV
    df = pd.read_csv(csv_path)

    # Randomly select 9 samples for visualization
    selected_samples = df.sample(n=9, random_state=99)

    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, selected_samples.iterrows()):
        try:
            # Construct image path and load the image
            img_dir = os.path.join(image_root, row['style'])
            img_path = os.path.join(img_dir, row['image_filename'])
            img = Image.open(img_path).convert("RGB")

            # Resize and crop the image to fit the target size
            img = ImageOps.fit(
                img,
                target_size,
                method=Image.Resampling.LANCZOS,
                bleed=0.2,
                centering=(0.5, 0.5)
            )
            img_arr = np.array(img)

            # Display the image and set the title
            ax.imshow(img_arr)
            ax.set_title(f"Style: {row['style']}", fontsize=18)
            ax.axis('off')

        except Exception as e:
            # Handle errors in loading or processing an image
            ax.axis('off')
            ax.text(
                0.5, 0.5,
                'Image load failed',
                horizontalalignment='center',
                verticalalignment='center'
            )

    # Adjust layout and save as vector PDF
    plt.tight_layout()
    fig.savefig(
        save_path,
        format='pdf',  # Export as vector PDF
        dpi=300,       # Control raster image resolution
        bbox_inches='tight'
    )
    plt.close(fig)
    print(f"Visualization saved as vector PDF to {save_path}")


if __name__ == "__main__":
    csv_path = "../../Datasets/Wiki/annotations.csv"
    image_root = "../../Datasets/Wiki"
    visualize_dataset(
        csv_path,
        image_root,
        save_path="dataset_samples.pdf",
        target_size=(256, 256)
    )