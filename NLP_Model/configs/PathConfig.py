from typing import List


class PathConfig:
    num_epochs: int = 10
    best_model_path: str = "best_clip.pth"
    demo_prompts: List[str] = [
        "Impressionist landscape with vibrant colors",
        "Cubist portrait with geometric shapes",
        "Abstract expressionist composition"
    ]