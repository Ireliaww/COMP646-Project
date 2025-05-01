# generate_with_ip_adapter.py

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS']    = '1'
import glob
import torch
from PIL import Image
from torchvision import transforms

import config
from ip_adapter import IPAdapter  # your adapter class that wraps Encoder+Decoder

# 1) Prepare device
DEVICE = torch.device("mps")

# 2) Pre- and post-processing (use same ImageNet norm as training)
FULL_RES_PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN,
                         std=config.IMAGENET_STD),
])
INVERSE    = config.INV_NORMALIZE

def stylize_with_adapter(adapter, content_path, style_path, out_path):
    """Run adapter on one content/style pair and save side by side."""
    c_img = Image.open(content_path).convert("RGB")
    s_img = Image.open(style_path).convert("RGB")

    # preprocess both images
    c_t = FULL_RES_PREPROCESS(c_img).unsqueeze(0).to(DEVICE)
    s_t = FULL_RES_PREPROCESS(s_img).unsqueeze(0).to(DEVICE)

    # forward through IP-Adapter
    with torch.no_grad():
        out_t = adapter.transfer(c_t, s_t)     # returns a normalized tensor [1,3,H,W]
        out_t = INVERSE(out_t.squeeze(0))      # undo ImageNet norm
        out_t = torch.clamp(out_t, 0, 1)

    # to PIL
    out_pil = Image.fromarray((out_t.cpu().numpy().transpose(1,2,0)*255).astype("uint8"))

    # resize stylized back to original resolution
    W, H = c_img.size
    out_pil = out_pil.resize((W, H), Image.LANCZOS)

    # now save two files: orig_XXX and styl_XXX
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base = os.path.basename(content_path)
    orig_out = os.path.join(os.path.dirname(out_path), f"orig_{base}")
    styl_out = os.path.join(os.path.dirname(out_path), f"styl_{base}")
    c_img.save(orig_out)
    out_pil.save(styl_out)
    print(f"Saved original → {orig_out}")
    print(f"Saved stylized → {styl_out}")

if __name__ == "__main__":
    # Loop over all content x style pairs you want to generate
    # you could also pick one style_file per style:
    for style_name in config.STYLES:
        # load adapter (loads both encoder & decoder under the hood)
        ckpt = os.path.join(config.OUT_DIR, f"decoder_{style_name}.pth")
        adapter = IPAdapter.load(ckpt, device=DEVICE)

        all_contents = glob.glob(os.path.join(config.CONTENT_DIR, "*.jpg"))
        # make one folder per style
        style_out_dir = os.path.join(config.IP_ADAPTER_OUT_DIR, style_name)
        os.makedirs(style_out_dir, exist_ok=True)
        # pick a representative style image
        style_example = glob.glob(
            os.path.join(config.STYLE_ROOT, style_name, "*.*")
            )[0]

        # stylize up to the first 5 content images
        for content_path in all_contents[:5]:
            # build your target file‐basename
            base = os.path.basename(content_path)
            # pass the _file_ path (without prefixes) to stylize_with_adapter
            out_path = os.path.join(style_out_dir, base)
            stylize_with_adapter(adapter, content_path, style_example, out_path)

