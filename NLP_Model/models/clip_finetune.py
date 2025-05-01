# clip_finetune.py
import math
import torch
import clip
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm

class CLIPFineTuner:
    """CLIP fine-tuning module with enhanced numerical stability, debugging support, and running epoch-loss tracking."""

    def __init__(self, device: str = "cuda"):
        """
        Load a pretrained CLIP model, freeze layers, initialize key params,
        reset and clamp logit_scale, enable anomaly detection.
        """
        self.device = device
        # Load CLIP and cast to float32
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model = self.model.float()

        # Freeze and init
        self._freeze_layers()
        self._safe_initialization()

        # logit_scale reset and clamp
        default_log_scale = math.log(1/0.07)
        self.min_log = math.log(1e-3)
        self.max_log = math.log(20.0)
        with torch.no_grad():
            self.model.logit_scale.data.fill_(default_log_scale)
            self.model.logit_scale.data.clamp_(self.min_log, self.max_log)

        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

    def _freeze_layers(self):
        """Freeze all visual encoder layers, unfreeze last text block."""
        for param in self.model.visual.parameters():
            param.requires_grad=False
        for name, param in self.model.transformer.named_parameters():
            param.requires_grad = ("resblocks.11" in name or "ln_final" in name)

    def _safe_initialization(self):
        """Initialize projection head and final norm layers."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "text_projection" in name and param.ndim==2:
                    nn.init.xavier_uniform_(param)
                elif name.endswith("ln_final.weight"):
                    nn.init.ones_(param)
                elif name.endswith("ln_final.bias"):
                    nn.init.zeros_(param)

    def _forward(self, images: torch.Tensor, texts: torch.Tensor):
        """Forward pass returning raw logits."""
        if images.dtype != torch.float32:
            images = images.float()
        return self.model(images, texts)

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler=None,
        epoch_idx: int=0
    ):
        """
        Train one epoch, tracking:
        - batch_losses: list of per-batch loss
        - running_epoch_losses: running average epoch loss per batch
        Returns:
            epoch_loss, batch_losses, running_epoch_losses
        """
        self.model.train()
        total_loss=0.0
        batch_count=0
        batch_losses=[]
        running_epoch_losses=[]

        progress = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}")
        for images, texts in progress:
            # skip invalid inputs
            if torch.isnan(images).any() or torch.isinf(images).any():
                continue
            images = images.to(self.device).float()
            texts = clip.tokenize(texts).to(self.device)

            optimizer.zero_grad()
            logits_img, logits_txt = self._forward(images, texts)
            # skip bad logits
            if torch.isnan(logits_img).any() or torch.isinf(logits_img).any():
                continue
            if torch.isnan(logits_txt).any() or torch.isinf(logits_txt).any():
                continue

            # compute scale
            scale = self.model.logit_scale.exp().clamp(1e-6,20.0)
            bs = images.size(0)
            labels = torch.arange(bs, device=self.device)
            logits_i = F.log_softmax(logits_img*scale,dim=-1)
            logits_t = F.log_softmax(logits_txt*scale,dim=-1)
            loss_i = F.nll_loss(logits_i,labels)
            loss_t = F.nll_loss(logits_t,labels)
            loss = 0.5*(loss_i+loss_t)
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler: scheduler.step()

            # update stats
            batch_losses.append(loss.item())
            total_loss+=loss.item()
            batch_count+=1
            running_avg = total_loss/batch_count
            running_epoch_losses.append(running_avg)

            progress.set_postfix({
                'batch_loss':f"{loss.item():.4f}",
                'epoch_loss':f"{running_avg:.4f}"}
            )

        epoch_loss = total_loss/max(batch_count,1)
        return epoch_loss, batch_losses, running_epoch_losses

    def dry_run(self,dataloader):
        self.model.eval()
        with torch.no_grad():
            for images, texts in dataloader:
                _ = self._forward(images.to(self.device).float(),clip.tokenize(texts).to(self.device))

    def save_model(self,save_path:str):
        torch.save(self.model.state_dict(), save_path)
