from diffusers import StableDiffusionPipeline
import torch


class StyleGenerator:
    """Style image generator using Stable Diffusion"""

    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to(device)

    def generate_image(self, prompt, **kwargs):
        """Generate style image from text prompt"""
        return self.pipe(
            prompt=prompt,
            num_inference_steps=kwargs.get('steps', 50),
            guidance_scale=kwargs.get('guidance', 7.5),
            height=512,
            width=512
        ).images[0]