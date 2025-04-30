import os
# must come _before_ any torch / numpy / sklearn / anything that pulls in MKL
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS']    = '1'
import glob
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image

import config
from model import VGGEncoder, Decoder, gram_matrix

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
# 1) Official VGG-19 preprocessing
preprocess = VGG19_Weights.IMAGENET1K_V1.transforms()

# 2) Style‐augmentation (random crops + flips + same normalization)
style_augment = transforms.Compose([
    transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    preprocess
])

# -----------------------------------------------------------------------------
# Dataset wrapper
# -----------------------------------------------------------------------------
class ListImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths     = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

# -----------------------------------------------------------------------------
# Compute per‐layer Gram targets for a given style folder
# -----------------------------------------------------------------------------
def compute_style_targets(style_folder):
    # collect all images
    exts = ("jpg","jpeg","png")
    style_paths = []
    for ext in exts:
        style_paths += glob.glob(os.path.join(style_folder, f"*.{ext}"))
    if not style_paths:
        raise RuntimeError(f"No style images found in {style_folder}")

    style_paths = style_paths[:50]

    # loader with augmentation
    ds     = ListImageDataset(style_paths, style_augment)
    loader = DataLoader(ds, config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    enc = VGGEncoder().to(DEVICE).eval()
    gram_sums = {}
    count     = 0

    with torch.no_grad():
        for batch in loader:
            x = batch.to(DEVICE)
            for idx, layer in enumerate(enc.enc_layers):
                x = layer(x)
                if idx in config.STYLE_LAYERS:
                    G = gram_matrix(x)            # [B, C, C]
                    if idx not in gram_sums:
                        gram_sums[idx] = torch.zeros_like(G[:1])
                    gram_sums[idx] += G.sum(dim=0, keepdim=True)
            count += x.size(0)

    # average
    return {l: gram_sums[l] / count for l in config.STYLE_LAYERS}

# -----------------------------------------------------------------------------
# Train one feed-forward decoder for a single style
# -----------------------------------------------------------------------------
def train_per_style(content_paths, gram_targets, save_path):
    content_paths = content_paths[:300]

    # content loader (no augmentation)
    ds     = ListImageDataset(content_paths, preprocess)
    loader = DataLoader(ds, config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    enc = VGGEncoder().to(DEVICE).eval()
    dec = Decoder().to(DEVICE)
    opt = torch.optim.Adam(dec.parameters(), lr=config.LR)

    best_loss = float("inf")
    best_ep   = 0

    for epoch in range(1, config.EPOCHS+1):
        running = 0.0

        for batch in loader:
            batch = batch.to(DEVICE)

            # content features (fixed)
            with torch.no_grad():
                f_c = enc(batch)

            # decode + losses
            x    = dec(f_c)
            f_x  = enc(x)
            loss_c = F.mse_loss(f_x, f_c)

            # style loss
            loss_s = 0.0
            y = batch
            for idx, layer in enumerate(enc.enc_layers):
                y = layer(y)
                if idx in config.STYLE_LAYERS:
                    Gx = gram_matrix(y)
                    Gs = gram_targets[idx].expand_as(Gx)
                    loss_s += F.mse_loss(Gx, Gs)

            # total variation
            tv = config.TV_WEIGHT * (
                (x[:,:,1:,:] - x[:,:,:-1,:]).abs().sum() +
                (x[:,:,:,1:] - x[:,:,:,:-1]).abs().sum()
            )

            loss = config.CONTENT_WEIGHT*loss_c + config.STYLE_WEIGHT*loss_s + tv

            # step
            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item() * batch.size(0)

        epoch_loss = running / len(ds)
        print(f"[{epoch}/{config.EPOCHS}] Loss: {epoch_loss:.4f}")

        # save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_ep   = epoch
            torch.save(dec.state_dict(), save_path)

    print(f"→ Best epoch {best_ep} with loss {best_loss:.4f}")

# -----------------------------------------------------------------------------
# Main loop: for each style in config.STYLES
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # gather content images
    content_paths = glob.glob(os.path.join(config.CONTENT_DIR, "*"))
    content_paths = [p for p in content_paths if p.lower().endswith(("jpg","jpeg","png"))]
    random.shuffle(content_paths)

    os.makedirs(config.OUT_DIR, exist_ok=True)

    for style in config.STYLES:
        style_folder = os.path.join(config.STYLE_ROOT, style)
        print(f"\n==> Computing style targets for '{style}'…")
        gram_t = compute_style_targets(style_folder)

        ckpt_path = os.path.join(config.OUT_DIR, f"decoder_{style}.pth")
        print(f"==> Training decoder for '{style}', saving to {ckpt_path}")
        train_per_style(content_paths, gram_t, ckpt_path)
