import os
import csv
import glob
import random
from itertools import cycle

# performance settings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS']    = '1'

import torch
# enable cudnn autotuner for optimized kernels
torch.backends.cudnn.benchmark = True
# single-threaded CPU for PyTorch
torch.set_num_threads(1)

torch.backends.openmp.enabled = False

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE       = 256
BATCH_SIZE     = 8
EPOCHS         = 6
LR             = 1e-4
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT   = 10.0
TV_WEIGHT      = 1e-6

# Indices of VGG layers for style loss (relu1_1, 2_1, 3_1, 4_1)
STYLE_LAYERS = [0, 5, 10, 19, 21]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def mul255(x):
    """Scale tensor from [0,1] to [0,255]."""
    return x * 255

class ListImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths     = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class VGGEncoder(nn.Module):
    """VGG19 up to relu4_1 for feature extraction."""
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.enc_layers = nn.Sequential(*list(vgg)[:22])
        for p in self.enc_layers.parameters():
            p.requires_grad = False
    def forward(self, x):
        return self.enc_layers(x)

class Decoder(nn.Module):
    """Mirror of encoder: maps 512×H/16×W/16 back to 3×H×W."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(512,256,3,1,1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,1,1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128,64,3,1,1),  nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64,3,3,1,1)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# Gram matrix for style loss
# -----------------------------------------------------------------------------
def gram_matrix(feat):
    b, c, h, w = feat.shape
    feat = feat.view(b, c, h * w)
    return torch.bmm(feat, feat.transpose(1, 2)) / (c * h * w)

# -----------------------------------------------------------------------------
# Precompute style Gram targets
# -----------------------------------------------------------------------------
def compute_style_targets(style_folder, transform, device):
    style_paths = glob.glob(os.path.join(style_folder, '*.jpg'))
    ds = ListImageDataset(style_paths, transform)
    loader = DataLoader(ds, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    encoder = VGGEncoder().to(device).eval()
    gram_sums = {l: 0.0 for l in STYLE_LAYERS}
    count = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x = batch
            for idx, layer in enumerate(encoder.enc_layers):
                x = layer(x)
                if idx in STYLE_LAYERS:
                    G = gram_matrix(x)
                    gram_sums[idx] += G.sum(dim=0, keepdim=True)
            count += batch.size(0)

    # average
    gram_avg = {l: (gram_sums[l] / count).to(device) for l in STYLE_LAYERS}
    return gram_avg

# -----------------------------------------------------------------------------
# Training per-style feed-forward network
# -----------------------------------------------------------------------------
def train_per_style(content_paths, gram_targets, transform, device):
    ds = ListImageDataset(content_paths, transform)
    loader = DataLoader(ds, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    encoder = VGGEncoder().to(device).eval()
    decoder = Decoder().to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        for batch in loader:
            batch = batch.to(device)

            # content features
            with torch.no_grad():
                f_c = encoder(batch)

            # decode from content features
            x = decoder(f_c)

            # content loss
            f_x = encoder(x)
            loss_c = F.mse_loss(f_x, f_c)

            # style loss
            loss_s = 0.0
            x_i = batch
            for idx, layer in enumerate(encoder.enc_layers):
                x_i = layer(x_i)
                if idx in STYLE_LAYERS:
                    Gx = gram_matrix(x_i)
                    Gs = gram_targets[idx].expand_as(Gx)
                    loss_s += F.mse_loss(Gx, Gs)

            # total variation
            loss_tv = TV_WEIGHT * (
                (x[:,:,1:,:]-x[:,:,:-1,:]).abs().sum() +
                (x[:,:,:,1:]-x[:,:,:,:-1]).abs().sum()
            )

            loss = CONTENT_WEIGHT*loss_c + STYLE_WEIGHT*loss_s + loss_tv
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}/{EPOCHS} → C {loss_c.item():.2f}  S {loss_s.item():.2f}")
        torch.save(decoder.state_dict(), f"decoder_style_epoch{epoch}.pth")

    return decoder

# -----------------------------------------------------------------------------
# Stylize helper (feed-forward only needs content)
# -----------------------------------------------------------------------------
def stylize(decoder, content_img_path, out_path, transform, device):
    decoder.eval()
    img = Image.open(content_img_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        f_c = VGGEncoder().to(device).eval()(inp)
        out = decoder(f_c).clamp(0,255) / 255.0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray((out.squeeze().cpu().numpy().transpose(1,2,0)*255).astype('uint8'))\
         .save(out_path)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # content images
    content_root = '../datasets/COCO/train2017'
    content_paths = [
        os.path.join(dp,fn)
        for dp,_,fns in os.walk(content_root)
        for fn in fns if fn.lower().endswith(('.jpg','png','jpeg'))
    ]
    random.shuffle(content_paths)
    c_split = 900
    train_content = content_paths[:c_split]
    val_content   = content_paths[c_split:1000]

    # style folder containing art-deco images
    style_folder = '../datasets/WikiArt/art-deco'

    # transform (no lambda inside DataLoader)
    TRANSFORM = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(mul255)
    ])

    print("Computing style targets…")
    gram_targets = compute_style_targets(style_folder, TRANSFORM, DEVICE)

    print("Training per-style network…")
    decoder = train_per_style(train_content, gram_targets, TRANSFORM, DEVICE)

    print("Evaluating content reconstruction loss…")
    enc = VGGEncoder().to(DEVICE).eval()
    tot = 0.0
    for img_path in val_content:
        img = TRANSFORM(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f_c = enc(img)
            out = decoder(f_c)
            tot += F.mse_loss(enc(out), f_c).item()
    print(f"Val content loss: {tot/len(val_content):.2f}")

    print("Stylizing examples…")
    os.makedirs('outputs', exist_ok=True)
    for i, img_path in enumerate(val_content[:5]):
        out_file = f"outputs/example_{i}.jpg"
        stylize(decoder, img_path, out_file, TRANSFORM, DEVICE)
        print("Saved", out_file)
