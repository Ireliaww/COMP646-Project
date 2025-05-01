import torch
import clip
from torch import nn, optim
from tqdm import tqdm


class CLIPFineTuner:
    """CLIP model fine-tuning module with NaN protection mechanisms"""

    def __init__(self, device="cuda"):
        self.device = device
        self.model, _ = clip.load("ViT-B/32", device=device)
        self._freeze_layers()
        self._safe_initialization()  # New initialization method

    def _freeze_layers(self):
        """Freeze non-critical layers for partial fine-tuning"""
        # Freeze visual encoder completely
        for param in self.model.visual.parameters():
            param.requires_grad = False

        # Only unfreeze last layers of text encoder
        for name, param in self.model.transformer.named_parameters():
            if "resblocks.11" in name or "ln_final" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _safe_initialization(self):
        """Safe parameter initialization for stability"""
        # Initialize final layer weights properly
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "text_projection" in name and param.ndim == 2:
                    nn.init.xavier_uniform_(param)
                elif "ln_final" in name and "weight" in name:
                    nn.init.ones_(param)
                elif "ln_final" in name and "bias" in name:
                    nn.init.zeros_(param)

    def _debug_forward(self, images, texts):
        """Safe forward pass with numerical checks"""
        # Forward pass with stability checks
        logits_per_image, logits_per_text = self.model(images, texts)

        # NaN/Inf protection
        if torch.isnan(logits_per_image).any() or torch.isinf(logits_per_image).any():
            logits_per_image = torch.nan_to_num(logits_per_image, nan=0.0, posinf=1e4, neginf=-1e4)

        if torch.isnan(logits_per_text).any() or torch.isinf(logits_per_text).any():
            logits_per_text = torch.nan_to_num(logits_per_text, nan=0.0, posinf=1e4, neginf=-1e4)

        return logits_per_image, logits_per_text

    def train_epoch(self, dataloader, optimizer, scheduler=None, epoch_idx=None):
        """Execute one training epoch with enhanced stability"""
        self.model.train()
        total_loss = 0.0
        batch_losses = []

        # Freeze logit scale parameter
        self.model.logit_scale.requires_grad_(False)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_idx + 1}")

        for batch_idx, (images, texts) in enumerate(progress_bar):
            # Data validation
            if torch.isnan(images).any() or torch.isinf(images).any():
                print("Invalid image data detected! Skipping batch...")
                continue

            # Move data to device
            images = images.to(self.device)
            texts = clip.tokenize(texts).to(self.device)

            try:
                # Safe forward pass
                logits_per_image, logits_per_text = self._debug_forward(images, texts)

                # Contrastive loss calculation
                batch_size = images.size(0)
                labels = torch.arange(batch_size).to(self.device)
                loss = (nn.CrossEntropyLoss()(logits_per_image, labels) +
                        nn.CrossEntropyLoss()(logits_per_text, labels)) / 2

                # Skip problematic batches
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Invalid loss value! Skipping batch...")
                    continue

                # Backpropagation with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                # Clamp logit scale for stability
                with torch.no_grad():
                    self.model.logit_scale.clamp_(-5, 5)

                # Record loss
                batch_loss = loss.item()
                total_loss += batch_loss
                batch_losses.append(batch_loss)

                # Update progress bar
                progress_bar.set_postfix({
                    'batch_loss': f'{batch_loss:.4f}',
                    'epoch_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue

        epoch_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        return epoch_loss, batch_losses

    def dry_run(self, dataloader):
        """Mock training method - forward passes only"""
        self.model.eval()
        with torch.no_grad():
            for images, texts in dataloader:
                images = images.to(self.device)
                texts = clip.tokenize(texts).to(self.device)
                _ = self._debug_forward(images, texts)
        print("Dry run completed without parameter updates")

    def save_model(self, save_path):
        """Save model weights"""
        torch.save(self.model.state_dict(), save_path)