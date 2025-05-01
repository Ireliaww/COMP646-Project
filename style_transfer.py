#!/usr/bin/env python3
import os, sys
# ─── 0) QUIET MKL WARNINGS ───────────────────────────────────────────────────
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS']      = '1'

from pathlib import Path
import torch
from PIL import Image

# ─── 1) PATH CONFIGURATION ───────────────────────────────────────────────────
ROOT          = Path(__file__).parent.resolve()
CONTENT_DIR   = ROOT / "example_contents"   # folder of your input images
MODEL_DIR     = ROOT / "fast_style_transfer_model"
CKPT_DIR      = MODEL_DIR / "checkpoints"

STYLE_SAMPLES = {
    "barock":                     ROOT / "NLP_Model" / "scripts" / "barock_style.png",
    "abstrakter-expressionismus": ROOT / "NLP_Model" / "scripts" / "abstrakter_expressionismus_style.png",
}

OUT_FF = ROOT / "generate_result"
OUT_IP = ROOT / "ip_adapter_result"

# ─── 2) DEVICE ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

# ─── 3) MAKE THE MODEL PACKAGE IMPORTABLE ────────────────────────────────────
# so that `from model import …` within those scripts still works
sys.path.insert(0, str(MODEL_DIR))

# ─── 4) IMPORT JUST THE HELPERS ──────────────────────────────────────────────
from fast_style_transfer_model.generate import load_decoder, load_encoder, stylize_image
from fast_style_transfer_model.generate_with_ip_adapter import load_adapter, stylize_with_adapter

# ─── 5) PICK FIRST 5 CONTENT IMAGES ───────────────────────────────────────────
all_imgs = sorted(
    [p for p in CONTENT_DIR.iterdir()
     if p.suffix.lower() in (".jpg","jpeg","png")]
)
first5 = all_imgs[:5]

# ─── 6) RUN EVERYTHING ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # ensure output roots exist
    OUT_FF.mkdir(exist_ok=True, parents=True)
    OUT_IP.mkdir(exist_ok=True, parents=True)

    for style_name, style_img in STYLE_SAMPLES.items():
        print(f"\n=== STYLE: '{style_name}' ===")

        # 1) load decoder & encoder
        print(" • loading decoder…", end=" ")
        # after (correct)
        decoder = load_decoder(style_name)

        decoder.to(DEVICE)
        print("ok")

        print(" • loading encoder…", end=" ")
        encoder = load_encoder()
        encoder.to(DEVICE)
        print("ok")

        # prepare per-style output dirs
        ff_out = OUT_FF / style_name
        ip_out = OUT_IP / style_name
        ff_out.mkdir(exist_ok=True, parents=True)
        ip_out.mkdir(exist_ok=True, parents=True)

        # 2) feed-forward generator
        print(" → feed-forward stylization:")
        for content_path in first5:
            img = Image.open(content_path).convert("RGB")
            stylized_img = stylize_image(decoder, encoder, img)

            output_file = ff_out / content_path.name
            stylized_img.save(output_file)
            print(f"  • Saved: {output_file}")

        # 3) IP-Adapter generator
        print(" → IP-Adapter stylization:")
        adapter = load_adapter(style_name)
        for content_path in first5:
            out_file = ip_out / content_path.name

            # Load content and style images
            content_img = Image.open(content_path).convert("RGB")
            style_image = Image.open(style_img).convert("RGB")

            # Stylize the content image
            stylized_img = stylize_with_adapter(adapter, content_img, style_image)

            # Save the resulting image
            stylized_img.save(out_file)
            print(f"  • Saved {out_file}")

    print("\n✅ All done!")
