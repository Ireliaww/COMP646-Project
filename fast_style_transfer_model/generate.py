import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS']    = '1'
import torch
from PIL import Image
from torchvision import transforms
from model import VGGEncoder, Decoder
import config
from pathlib import Path

# ─── 1) DEVICE ───────────────────────────────────────────────────────────────
DEVICE = torch.device("mps")

# ─── 2) IMAGENET STATS & TRANSFORMS ──────────────────────────────────────────
# These must match exactly what you used during training.
IMAGENET_MEAN = config.IMAGENET_MEAN
IMAGENET_STD  = config.IMAGENET_STD
IMG_SIZE      = config.IMG_SIZE

PREPROCESS = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

INV_NORMALIZE = transforms.Normalize(
    mean=[-m/s for m,s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[ 1.0/s for   s in IMAGENET_STD]
)

# ─── 3) MODEL LOADERS ────────────────────────────────────────────────────────
def load_encoder():
    enc = VGGEncoder().to(DEVICE)
    enc.eval()
    return enc

def load_decoder(style_name):
    MODEL_DIR = Path(__file__).parent.resolve()
    ckpt = MODEL_DIR / "checkpoints" / f"decoder_{style_name}.pth"
    dec = Decoder()
    dec.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    dec.eval()
    return dec

# ─── 4) STYLIZE ONE IMAGE ────────────────────────────────────────────────────
def stylize_image(decoder, encoder, img: Image.Image) -> Image.Image:
    """Returns a PIL image of the stylized result."""
    inp = PREPROCESS(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = encoder(inp)
        raw   = decoder(feats).squeeze(0)      # still normalized
        out   = INV_NORMALIZE(raw)             # undo normalization
        out   = torch.clamp(out, 0, 1)
    arr = (out.cpu().numpy().transpose(1,2,0) * 255).astype("uint8")
    return Image.fromarray(arr)

# ─── 5) MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # load encoder just once
    encoder = load_encoder()

    # gather and sort all content images, then take the first 5
    all_imgs = sorted(glob.glob(os.path.join(config.CONTENT_DIR, "*")))
    all_imgs = [p for p in all_imgs
                if p.lower().endswith((".jpg","jpeg","png"))]
    first5 = all_imgs[:5]

    # for each style, load its decoder and run through the 5 images
    for style in config.STYLES:
        print(f"\n→ Generating style: {style!r}")
        decoder = load_decoder(style)

        out_dir = os.path.join(config.GENERATE_OUT_DIR, style)
        os.makedirs(out_dir, exist_ok=True)

        for path in first5:
            base = os.path.basename(path)
            orig = Image.open(path).convert("RGB")
            styl = stylize_image(decoder, encoder, orig)

            # save side-by-side files
            orig_out = os.path.join(out_dir,  f"orig_{base}")
            styl_out = os.path.join(out_dir, f"styl_{style}_{base}")

            orig .save(orig_out)
            styl .save(styl_out)
            print(f"  • Saved {orig_out}")
            print(f"    Saved {styl_out}")