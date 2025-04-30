import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import VGGEncoder, Decoder
from torchvision.models import VGG19_Weights


def load_decoder(path, device):
    """Load a trained Decoder checkpoint."""
    dec = Decoder().to(device)
    state = torch.load(path, map_location=device)
    dec.load_state_dict(state)
    dec.eval()
    return dec

# official ImageNet preprocess (matches model training)
PREPROCESS = VGG19_Weights.IMAGENET1K_V1.transforms()
# inverse normalize to [0..1]
IMAGENET_MEAN = VGG19_Weights.IMAGENET1K_V1.meta['mean'] if 'meta' in dir(VGG19_Weights.IMAGENET1K_V1) \
                else VGG19_Weights.IMAGENET1K_V1.transforms()._tfms[2].mean
IMAGENET_STD  = VGG19_Weights.IMAGENET1K_V1.meta['std']  if 'meta' in dir(VGG19_Weights.IMAGENET1K_V1) \
                else VGG19_Weights.IMAGENET1K_V1.transforms()._tfms[2].std
INV_NORMALIZE = transforms.Normalize(
    mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[1/s for s in IMAGENET_STD]
)


def stylize(decoder, content_path, output_path, device):
    """Stylize a single content image and save side-by-side with original."""
    # load & preprocess
    img = Image.open(content_path).convert('RGB')
    inp = PREPROCESS(img).unsqueeze(0).to(device)

    # encode + decode
    encoder = VGGEncoder().to(device).eval()
    with torch.no_grad():
        f_c = encoder(inp)
        raw = decoder(f_c)
        out = INV_NORMALIZE(raw.squeeze(0))
        out = torch.clamp(out, 0, 1)

    # convert to PIL
    stylized = Image.fromarray((out.cpu().numpy().transpose(1,2,0)*255).astype('uint8'))
    base = os.path.basename(content_path)
    # ensure output
    os.makedirs(output_path, exist_ok=True)
    orig_out = os.path.join(output_path, f"orig_{base}")
    styl_out = os.path.join(output_path, f"styl_{base}")
    img.save(orig_out)
    stylized.save(styl_out)
    print(f"Saved: {orig_out}, {styl_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate stylized images using a trained Decoder.")
    parser.add_argument('--model',    type=str, required=True, help='Path to the .pth decoder checkpoint')
    parser.add_argument('--content',  type=str, required=True, help='Path to a content image or folder')
    parser.add_argument('--out_dir',  type=str, default='outputs', help='Directory to save results')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    decoder = load_decoder(args.model, device)

    # if content is a folder, process all images inside
    paths = []
    if os.path.isdir(args.content):
        for fn in os.listdir(args.content):
            if fn.lower().endswith(('.jpg','jpeg','png')):
                paths.append(os.path.join(args.content, fn))
    else:
        paths = [args.content]

    for p in paths:
        stylize(decoder, p, args.out_dir, device)
