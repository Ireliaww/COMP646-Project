from clip import clip
from NLP_Model.configs.paths import PathConfig
from NLP_Model.models.clip_finetune import CLIPFineTuner
from NLP_Model.models.sd_generator import StyleGenerator
import torch


def generate_style_reference(prompt):
    """End-to-end style reference generation"""
    # Load configuration
    config = PathConfig()

    # Initialize models
    clip_model = CLIPFineTuner(device="cuda")
    clip_model.model.load_state_dict(torch.load(config.finetuned_model_path))
    generator = StyleGenerator()

    # Get text embedding
    text_input = clip.tokenize([prompt]).to("cuda")
    with torch.no_grad():
        embedding = clip_model.model.encode_text(text_input)

    # Generate image
    image = generator.generate_image(prompt)
    return image, embedding


if __name__ == "__main__":
    user_prompt = "A painting in the {x} style, professional artwork, trending on artstation"
    style_image, style_embedding = generate_style_reference(user_prompt)
    style_image.save("generated_style.png")