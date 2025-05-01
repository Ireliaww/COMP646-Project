# Path configurations for dataset and outputs
class PathConfig:
    def __init__(self):
        self.csv_path = "../../Datasets/Wiki/annotations.csv"
        self.image_root = "../../Datasets/Wiki"
        self.output_dir = "./outputs"
        self.finetuned_model_path = "../models/clip_finetuned.pth"