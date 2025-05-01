import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

# -----------------------------------------------------------------------------
# Encoder / Decoder definitions & helper fns
# -----------------------------------------------------------------------------

class VGGEncoder(nn.Module):
    """VGG19 up to relu4_1 for feature extraction."""
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # take layers [:22] ⇒ up through relu4_1
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

def adaptive_instance_norm(content_feat, style_feat, eps=1e-5):
    """AdaIN: aligns mean & variance of content to style."""
    c_mean = content_feat.mean([2,3], keepdim=True)
    c_std  = content_feat.std([2,3], keepdim=True) + eps
    s_mean = style_feat.mean([2,3], keepdim=True)
    s_std  = style_feat.std([2,3], keepdim=True) + eps
    return s_std * (content_feat - c_mean) / c_std + s_mean

def gram_matrix(feat):
    """Compute Gram matrix for style loss."""
    b, c, h, w = feat.shape
    F = feat.view(b, c, h*w)
    return torch.bmm(F, F.transpose(1,2)) / (c * h * w)
