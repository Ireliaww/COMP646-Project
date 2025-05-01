# evaluate.py

import os
import glob
import random

# must come _before_ any torch / numpy / sklearn / anything that pulls in MKL
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS']    = '1'

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import VGG19_Weights
from torchvision import transforms
from PIL import Image

import config
from model import VGGEncoder, Decoder, gram_matrix

# ─── Device ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Preprocessing ─────────────────────────────────────────────────────────
PREPROCESS = VGG19_Weights.IMAGENET1K_V1.transforms()

# ─── A top‐level Dataset for style images (picklable) ────────────────────────
class StyleDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

# ─── Compute style Gram‐matrix targets ──────────────────────────────────────
def compute_style_targets(style_folder):
    """Compute average Gram matrices at each STYLE_LAYERS over all style images."""
    # gather style image paths
    exts = ("jpg","jpeg","png")
    style_paths = []
    for ext in exts:
        style_paths += glob.glob(os.path.join(style_folder, f"*.{ext}"))
    if not style_paths:
        raise FileNotFoundError(f"No style images in {style_folder!r}")

    # loader with augmentation or just PREPROCESS if you prefer no aug here
    ds = StyleDataset(style_paths, PREPROCESS)
    loader = DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    enc = VGGEncoder().to(DEVICE).eval()
    gram_sums = {}  # layer → [1,C,C]
    count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch.to(DEVICE)
            for idx, layer in enumerate(enc.enc_layers):
                x = layer(x)
                if idx in config.STYLE_LAYERS:
                    G = gram_matrix(x)               # [B,C,C]
                    if idx not in gram_sums:
                        gram_sums[idx] = torch.zeros_like(G[:1])
                    gram_sums[idx] += G.sum(dim=0, keepdim=True)
            count += x.size(0)

    # average and return
    return {l: gram_sums[l] / count for l in config.STYLE_LAYERS}


# ─── Build held‐out content split ───────────────────────────────────────────
all_contents = glob.glob(os.path.join(config.CONTENT_DIR, "*"))
all_contents = [
    p for p in all_contents
    if p.lower().endswith((".jpg","jpeg","png"))
]
all_contents.sort()
random.seed(0)
random.shuffle(all_contents)

TRAIN_CONTENT_NUM = 500
VAL_CONTENT_NUM   = 50
val_paths = all_contents[
    TRAIN_CONTENT_NUM : TRAIN_CONTENT_NUM + VAL_CONTENT_NUM
]
print(f"Evaluating on {len(val_paths)} held‐out content images "
      f"(indices {TRAIN_CONTENT_NUM}…"
      f"{TRAIN_CONTENT_NUM + VAL_CONTENT_NUM - 1})\n")


# ─── Main evaluation ──────────────────────────────────────────────────────
def evaluate_style(style_name):
    style_folder = os.path.join(config.STYLE_ROOT, style_name)
    print(f"∘ Style = {style_name!r}: computing targets…", end=" ")
    style_targets = compute_style_targets(style_folder)
    print("done")

    ckpt_path = os.path.join(config.OUT_DIR, f"decoder_{style_name}.pth")
    print(f"∘ Loading checkpoint: {ckpt_path}…", end=" ")
    dec = Decoder().to(DEVICE)
    dec.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    dec.eval()
    print("done\n")

    enc = VGGEncoder().to(DEVICE).eval()

    tot_c = 0.0
    tot_s = 0.0
    n     = 0

    for cp in val_paths:
        img = Image.open(cp).convert("RGB")
        t   = PREPROCESS(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # content reconstruction
            f_c   = enc(t)
            x     = dec(f_c)
            f_x   = enc(x)
            loss_c = F.mse_loss(f_x, f_c).item()

            # style loss
            loss_s = 0.0
            y = t
            for idx, layer in enumerate(enc.enc_layers):
                y = layer(y)
                if idx in config.STYLE_LAYERS:
                    Gx = gram_matrix(y)
                    Gs = style_targets[idx].expand_as(Gx)
                    loss_s += F.mse_loss(Gx, Gs).item()

        tot_c += loss_c
        tot_s += loss_s
        n     += 1

    print(f"→ {style_name}: "
          f"Avg Content Loss = {tot_c/n:.4f}, "
          f"Avg Style Loss   = {tot_s/n:.4f}\n")


if __name__ == "__main__":
    for style in config.STYLES:
        evaluate_style(style)
