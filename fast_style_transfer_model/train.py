# train.py

import os
import glob
import random

# must come _before_ any torch / numpy / sklearn / anything that pulls in MKL
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS']    = '1'

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image

import config
from model import (
    VGGEncoder,
    Decoder,
    gram_matrix,
    adaptive_instance_norm,    # ← make sure AdaIN is available
)

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
DEVICE = torch.device("mps")

# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
preprocess = VGG19_Weights.IMAGENET1K_V1.transforms()

content_augment = transforms.Compose([
    transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])

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
    # collect style images
    exts = ("jpg","jpeg","png")
    style_paths = []
    for ext in exts:
        style_paths += glob.glob(os.path.join(style_folder, f"*.{ext}"))
    if not style_paths:
        raise RuntimeError(f"No style images found in {style_folder}")
    style_paths = style_paths[:200]

    ds     = ListImageDataset(style_paths, style_augment)
    loader = DataLoader(ds, config.BATCH_SIZE,
                        shuffle=False, num_workers=4, pin_memory=True)

    enc       = VGGEncoder().to(DEVICE).eval()
    gram_sums = {}  # will hold sum of G per layer
    count     = 0

    with torch.no_grad():
        for batch in loader:
            x = batch.to(DEVICE)
            for idx, layer in enumerate(enc.enc_layers):
                x = layer(x)
                if idx in config.STYLE_LAYERS:
                    G = gram_matrix(x)         # [B, C, C]
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
    content_paths = content_paths[:500]

    ds     = ListImageDataset(content_paths, content_augment)
    loader = DataLoader(ds, config.BATCH_SIZE,
                        shuffle=True, num_workers=8, pin_memory=True)

    enc       = VGGEncoder().to(DEVICE).eval()
    dec       = Decoder().to(DEVICE)
    optimizer = torch.optim.Adam(dec.parameters(), lr=config.LR)

    best_loss = float("inf")
    best_ep   = 0

    for epoch in range(1, config.EPOCHS+1):
        running = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            with torch.no_grad():
                f_c = enc(batch)
            x      = dec(f_c)
            f_x    = enc(x)
            loss_c = F.mse_loss(f_x, f_c)

            loss_s = 0.0
            y      = batch
            for idx, layer in enumerate(enc.enc_layers):
                y = layer(y)
                if idx in config.STYLE_LAYERS:
                    Gx = gram_matrix(y)
                    Gs = gram_targets[idx].expand_as(Gx)
                    loss_s += F.mse_loss(Gx, Gs)

            tv   = config.TV_WEIGHT * (
                (x[:,:,1:,:] - x[:,:,:-1,:]).abs().sum() +
                (x[:,:,:,1:] - x[:,:,:,:-1]).abs().sum()
            )
            loss = config.CONTENT_WEIGHT*loss_c + config.STYLE_WEIGHT*loss_s + tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * batch.size(0)

        epoch_loss = running / len(ds)
        print(f"[{epoch}/{config.EPOCHS}] Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss, best_ep = epoch_loss, epoch
            torch.save(dec.state_dict(), save_path)

    print(f"→ Best epoch {best_ep} with loss {best_loss:.4f}")
    return dec

# -----------------------------------------------------------------------------
# Stylize and save one full-res example (side-by-side)
# -----------------------------------------------------------------------------
FULL_RES_PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])
INV_NORMALIZE = transforms.Normalize(
    mean=[-m/s for m,s in zip([0.485,0.456,0.406],
                              [0.229,0.224,0.225])],
    std =[1/s   for s   in           [0.229,0.224,0.225]]
)

def save_example(decoder, encoder, content_path, style_path, out_path):
    orig = Image.open(content_path).convert("RGB")
    W, H = orig.size
    style = Image.open(style_path).convert("RGB")

    c_t = FULL_RES_PREPROCESS(orig).unsqueeze(0).to(DEVICE)
    s_t = FULL_RES_PREPROCESS(style).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        f_c = encoder(c_t)
        f_s = encoder(s_t)
        t   = adaptive_instance_norm(f_c, f_s)
        out = decoder(t).squeeze(0)
        out = INV_NORMALIZE(out)
        out = torch.clamp(out, 0.0, 1.0)

    arr      = (out.cpu().numpy().transpose(1,2,0)*255).astype("uint8")
    stylized = Image.fromarray(arr).resize((W,H), Image.LANCZOS)

    canvas = Image.new("RGB", (W*2, H))
    canvas.paste(orig,     (0,0))
    canvas.paste(stylized, (W,0))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    print(f"Saved example to {out_path}")

# -----------------------------------------------------------------------------
# Main loop: train & save one example per style
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    content_paths = glob.glob(os.path.join(config.CONTENT_DIR, "*"))
    content_paths = [p for p in content_paths
                     if p.lower().endswith(("jpg","jpeg","png"))]
    random.shuffle(content_paths)

    os.makedirs(config.OUT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLE_OUT_DIR, exist_ok=True)

    for style in config.STYLES:
        style_folder = os.path.join(config.STYLE_ROOT, style)
        print(f"\n==> Computing style targets for '{style}'…")
        gram_t   = compute_style_targets(style_folder)

        ckpt_path = os.path.join(config.OUT_DIR, f"decoder_{style}.pth")
        print(f"==> Training decoder for '{style}', saving to {ckpt_path}")
        decoder = train_per_style(content_paths, gram_t, ckpt_path)

        # pick one content + one style example
        content_example = 'example.jpg'
        style_example   = glob.glob(os.path.join(style_folder,"*.jpg"))[0]
        out_path        = os.path.join(config.SAMPLE_OUT_DIR,
                                       f"{style}_example.jpg")

        print(f"==> Writing example for '{style}'…")
        save_example(decoder,
                     VGGEncoder().to(DEVICE).eval(),
                     content_example,
                     style_example,
                     out_path)
