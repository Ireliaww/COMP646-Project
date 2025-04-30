# utils/logger.py

import os
import logging
from datetime import datetime
from typing import Dict, Any
import torch
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """Unified logging system with multiple backends"""

    def __init__(self,
                 log_dir: str = "logs",
                 experiment_name: str = "clip_finetune",
                 use_tensorboard: bool = True,
                 use_wandb: bool = False):
        """
        Initialize logging system

        Args:
            log_dir: Root directory for all logs
            experiment_name: Identifier for current experiment
            use_tensorboard: Enable TensorBoard logging
            use_wandb: Enable Weights & Biases logging
        """
        self.log_dir = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize console logger
        self.console_logger = logging.getLogger(experiment_name)
        self.console_logger.setLevel(logging.INFO)
        self._setup_console_logger()

        # Initialize TensorBoard
        self.tb_writer = SummaryWriter(self.log_dir) if use_tensorboard else None

        # Initialize W&B
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb.init(project=experiment_name)
            except ImportError:
                self.console_logger.warning("W&B not installed, skipping W&B logging")

    def _setup_console_logger(self):
        """Configure console logging handler"""
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.console_logger.addHandler(ch)

    def log_metrics(self,
                    metrics: Dict[str, Any],
                    step: int,
                    prefix: str = "train"):
        """
        Log metrics to all enabled backends

        Args:
            metrics: Dictionary of metric values
            step: Current step/epoch number
            prefix: Namespace prefix for metrics
        """
        # Console logging
        self.console_logger.info(f"Step {step} | {prefix} metrics: {metrics}")

        # TensorBoard logging
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"{prefix}/{key}", value, step)

        # W&B logging
        if self.wandb:
            self.wandb.log({f"{prefix}_{key}": value for key, value in metrics.items()}, step=step)

    def log_hyperparams(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        if self.tb_writer:
            self.tb_writer.add_hparams(params, {})
        if self.wandb:
            self.wandb.config.update(params)

    def save_checkpoint(self,
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        epoch: int,
                        filename: str = "checkpoint.pth"):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(self.log_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        self.console_logger.info(f"Checkpoint saved at {checkpoint_path}")

    def close(self):
        """Cleanup logging resources"""
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb:
            self.wandb.finish()