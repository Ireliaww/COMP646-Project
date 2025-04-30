import argparse
from typing import List

from torch import nn, optim

from NLP_Model.configs.paths import PathConfig
from NLP_Model.data.dataset import ArtStyleDataset
from NLP_Model.data.preprocessing import get_clip_transform
from NLP_Model.models.clip_finetune import CLIPFineTuner
from torch.utils.data import DataLoader
# Add to imports
from NLP_Model.utils.logger import ExperimentLogger
from NLP_Model.utils.metrics import CLIPMetrics
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_curve(train_loss_history: List[float], save_path: str = None):
    """Visualize training loss progression"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, marker='o', linestyle='-', color='b')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def main():
    # Configuration
    config = PathConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize components
    transform = get_clip_transform(train=True)
    dataset = ArtStyleDataset(config.csv_path, config.image_root, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Training setup
    trainer = CLIPFineTuner(device="cuda")
    optimizer = optim.AdamW(trainer.model.parameters(), lr=5e-6)
    logger = ExperimentLogger()

    # Training loop
    num_epochs = 1
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss, _ = trainer.train_epoch(dataloader, optimizer, epoch_idx=epoch)
        loss_history.append(epoch_loss)
        logger.log_metrics({"epoch_loss": epoch_loss}, epoch)

    # Visualization
    plot_training_curve(loss_history,
                        save_path=os.path.join(config.output_dir, "training_curve.png"))
    # Save model
    trainer.save_model(config.finetuned_model_path)

if __name__ == "__main__":
    main()