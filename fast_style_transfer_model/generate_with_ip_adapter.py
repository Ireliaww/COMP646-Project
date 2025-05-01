# generate_with_ip_adapter.py

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS']    = '1'

import glob
import torch
from PIL import Image
from torchvision import transforms

import config
from model import VGGEncoder, Decoder, adaptive_instance_norm
from ip_adapter import IPAdapter
from pathlib import Path

# ─── 1) DEVICE ───────────────────────────────────────────────────────────────
DEVICE = torch.device("mps")

# ─── 2) PRE-/POST‐PROCESSING ─────────────────────────────────────────────────────
PREPROCESS = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.CenterCrop(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])
INV_NORMALIZE = config.INV_NORMALIZE

# ─── 3) BUILD & LOAD AN IP-ADAPTER ──────────────────────────────────────────────
def load_adapter(style_name):
    MODEL_DIR = Path(__file__).parent.resolve()

    enc = VGGEncoder().to(DEVICE).eval()
    dec = Decoder().to(DEVICE)

    ckpt = MODEL_DIR / "checkpoints" / f"decoder_{style_name}.pth"
    dec.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    dec.eval()

    return IPAdapter(enc, dec)


# ─── 4) STYLIZE ONE PAIR ────────────────────────────────────────────────────────
def stylize_with_adapter(adapter, content_img, style_img):
    # preprocess
    c_t = PREPROCESS(content_img).unsqueeze(0).to(DEVICE)
    s_t = PREPROCESS(style_img).unsqueeze(0).to(DEVICE)
    # forward
    with torch.no_grad():
        out_t = adapter.transfer(c_t, s_t)        # [1,3,H,W] normalized
        out_t = INV_NORMALIZE(out_t.squeeze(0))    # undo ImageNet norm
        out_t = torch.clamp(out_t, 0, 1)
    # to PIL
    arr = (out_t.cpu().numpy().transpose(1,2,0) * 255).astype("uint8")
    return Image.fromarray(arr)


# ─── 5) MAIN LOOP ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # pick first 5 content images
    all_contents = sorted(glob.glob(os.path.join(config.CONTENT_DIR, "*")))
    all_contents = [p for p in all_contents if p.lower().endswith((".jpg","jpeg","png"))][:5]

    for style in config.STYLES:
        print(f"\n→ Generating style: {style!r}")
        adapter = load_adapter(style)

        # pick one representative style image
        style_folder = os.path.join(config.STYLE_ROOT, style)
        style_example = glob.glob(os.path.join(style_folder, "*"))[0]
        style_img = Image.open(style_example).convert("RGB")

        # make output dir for this style
        out_dir = os.path.join(config.IP_ADAPTER_OUT_DIR, style)
        os.makedirs(out_dir, exist_ok=True)

        # process each of the 5
        for content_path in all_contents:
            base = os.path.basename(content_path)
            content_img = Image.open(content_path).convert("RGB")

            styl_img = stylize_with_adapter(adapter, content_img, style_img)

            # save both
            orig_out = os.path.join(out_dir, f"orig_{base}")
            styl_out = os.path.join(out_dir, f"styl_{style}_{base}")
            content_img.save(orig_out)
            styl_img.save(styl_out)

            print(f"  • Saved {orig_out}")
            print(f"    Saved {styl_out}")
