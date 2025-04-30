import torch
import clip
from torch import nn, optim
from tqdm import tqdm


class CLIPFineTuner:
    """CLIP model fine-tuning module"""

    def __init__(self, device="cuda"):
        self.device = device
        self.model, _ = clip.load("ViT-B/32", device=device)
        self._freeze_layers()

    def _freeze_layers(self):
        """Freeze non-critical layers for partial fine-tuning"""
        # Freeze visual encoder
        for param in self.model.visual.parameters():
            param.requires_grad = False

        # Unfreeze last layers of text encoder
        for name, param in self.model.transformer.named_parameters():
            if "resblocks.11" in name or "ln_final" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def train_epoch(self, dataloader, optimizer, scheduler=None, epoch_idx=None):
        """Execute one training epoch with loss tracking"""
        self.model.train()
        total_loss = 0.0
        batch_losses = []

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_idx + 1}")

        for batch_idx, (images, texts) in enumerate(progress_bar):
            # Data transfer
            images = images.to(self.device)
            texts = clip.tokenize(texts).to(self.device)

            # Forward pass
            logits_per_image, logits_per_text = self.model(images, texts)

            # Loss calculation
            batch_size = images.size(0)
            labels = torch.arange(batch_size).to(self.device)
            loss = (nn.CrossEntropyLoss()(logits_per_image, labels) +
                    nn.CrossEntropyLoss()(logits_per_text, labels)) / 2

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Loss recording
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)

            # Progress update
            progress_bar.set_postfix({
                'batch_loss': f'{batch_loss:.4f}',
                'epoch_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })

            if scheduler:
                scheduler.step()

        epoch_loss = total_loss / len(dataloader)
        return epoch_loss, batch_losses

    def save_model(self, save_path):
        """Save model weights"""
        torch.save(self.model.state_dict(), save_path)