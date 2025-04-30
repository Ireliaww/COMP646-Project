# Path configurations for dataset and outputs
class PathConfig:
    def __init__(self):
        self.csv_path = "../../Datasets/WikiArt/annotations.csv"
        self.image_root = "../../Datasets/WikiArt"
        self.output_dir = "./outputs"
        self.finetuned_model_path = "../models/clip_finetuned.pth"