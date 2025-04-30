import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

# -- 1) AdaIN function -----------------------------------------------
def adaptive_instance_norm(content_feat, style_feat, eps=1e-5):
    # channel-wise mean & std
    c_mean = content_feat.mean([2,3], keepdim=True)
    c_std  = content_feat.std([2,3], keepdim=True) + eps
    s_mean = style_feat.mean([2,3], keepdim=True)
    s_std  = style_feat.std([2,3], keepdim=True) + eps
    return s_std * (content_feat - c_mean) / c_std + s_mean

# -- 2) Encoder (VGG19 upto relu4_1) --------------------------------
class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        # slice upto relu4_1 (layer 21)
        self.enc_layers = nn.Sequential(*[vgg[x] for x in range(22)])
        for p in self.enc_layers.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.enc_layers(x)

# -- 3) Decoder (mirror of encoder) ----------------------------------
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # This mirror architecture may need tuning to match the encoder's channels.
        self.net = nn.Sequential(
            # input: 512 x H/16 x W/16
            nn.Conv2d(512,256,3,1,1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),    # ->256 x H/8
            nn.Conv2d(256,256,3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,1,1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),    # ->128 x H/4
            nn.Conv2d(128,128,3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,1,1),  nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),    # ->64 x H/2
            nn.Conv2d(64,64,3,1,1),   nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),    # ->64 x H
            nn.Conv2d(64,3,3,1,1)
        )

    def forward(self, x):
        return self.net(x)

# -- 4) Gram matrix --------------------------------------------------
def gram_matrix(feat):
    (b, c, h, w) = feat.size()
    feat = feat.view(b, c, h*w)
    return torch.bmm(feat, feat.transpose(1,2)) / (c * h * w)

# -- 5) Training pipeline --------------------------------------------
def train_fast_style_transfer(
    content_dir, style_dir,
    batch_size=8, img_size=256,
    epochs=2, lr=1e-4,
    content_weight=1.0, style_weight=10.0, tv_weight=1e-6,
    device='cuda'
):
    # a) Data loaders
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])
    content_ds = datasets.ImageFolder(content_dir, transform=tf)
    style_ds   = datasets.ImageFolder(style_dir,   transform=tf)
    content_loader = DataLoader(content_ds, batch_size, shuffle=True, num_workers=4)
    style_loader   = DataLoader(style_ds,   batch_size, shuffle=True, num_workers=4)
    style_iter = iter(style_loader)

    # b) Models
    encoder = VGGEncoder().to(device).eval()
    decoder = Decoder().to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    # c) Loss network layers for style loss
    # we will extract features at relu1_1, relu2_1, relu3_1, relu4_1
    style_layers = [0, 5, 10, 19, 21]  # indices in VGG features

    for epoch in range(epochs):
        for i, (content_batch, _) in enumerate(content_loader):
            try:
                style_batch, _ = next(style_iter)
            except StopIteration:
                style_iter = iter(style_loader)
                style_batch, _ = next(style_iter)

            content_batch = content_batch.to(device)
            style_batch   = style_batch.to(device)

            # forward
            f_c = encoder(content_batch)
            f_s = encoder(style_batch)
            t   = adaptive_instance_norm(f_c, f_s)
            x   = decoder(t)

            # compute losses
            # 1) content loss on relu4_1: Enc(x) vs t
            f_x = encoder(x)
            loss_c = F.mse_loss(f_x, t)

            # 2) style loss: sum over layers
            loss_s = 0.0
            feats_x = [encoder.enc_layers[:l+1](x) for l in style_layers]
            feats_s = [encoder.enc_layers[:l+1](style_batch) for l in style_layers]
            for fx, fs in zip(feats_x, feats_s):
                loss_s += F.mse_loss(gram_matrix(fx), gram_matrix(fs))

            # 3) total variation loss
            loss_tv = tv_weight * (
                torch.sum(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])) +
                torch.sum(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]))
            )

            loss = content_weight * loss_c + style_weight * loss_s + loss_tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{i+1}/{len(content_loader)}] "
                      f"Loss: {loss.item():.4f} (C {loss_c:.4f}, S {loss_s:.4f})")

        # save checkpoint each epoch
        torch.save(decoder.state_dict(), f"decoder_epoch{epoch+1}.pth")

    return decoder

# -- 6) Inference ----------------------------------------------------
def stylize(decoder, content_img_path, style_img_path, out_path,
            img_size=512, device='cuda'):
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])
    loader = lambda p: tf(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
    c = loader(content_img_path)
    s = loader(style_img_path)

    encoder = VGGEncoder().to(device).eval()
    decoder = decoder.to(device).eval()

    with torch.no_grad():
        f_c = encoder(c)
        f_s = encoder(s)
        t   = adaptive_instance_norm(f_c, f_s)
        out = decoder(t).clamp(0, 255) / 255.0

    # save output
    out_img = transforms.ToPILImage()(out.squeeze().cpu())
    out_img.save(out_path)
    print(f"Saved stylized image to {out_path}")

# -- 7) Example usage -----------------------------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Train (you’ll want more epochs in practice—e.g. 4–6)
    decoder = train_fast_style_transfer(
        content_dir='../datasets/COCO/train2017',
        style_dir  ='../datasets/WikiArt/wikiart/images',
        batch_size = 8,
        img_size   = 256,
        epochs     = 4,
        lr         = 1e-4,
        device     = device
    )
    # Inference
    stylize(decoder,
            content_img_path='examples/content.jpg',
            style_img_path  ='examples/style.jpg',
            out_path        ='examples/output.jpg',
            img_size=512,
            device=device)
