# ip_adapter.py
import torch
import torch.nn.functional as F
from model import VGGEncoder, Decoder, adaptive_instance_norm

class IPAdapter:
    def __init__(self, encoder: VGGEncoder, decoder: Decoder):
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def load(cls, ckpt_path: str, device: torch.device):
        # instantiate clean models
        enc = VGGEncoder().to(device).eval()
        dec = Decoder().to(device)
        # load only the decoder weights
        state = torch.load(ckpt_path, map_location=device)
        dec.load_state_dict(state)
        dec.eval()
        return cls(enc, dec)

    def transfer(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Given preprocessed content & style [1,3,H,W], returns stylized [1,3,H,W]."""
        with torch.no_grad():
            f_c = self.encoder(content)
            f_s = self.encoder(style)
            t   = adaptive_instance_norm(f_c, f_s)
            return self.decoder(t)
