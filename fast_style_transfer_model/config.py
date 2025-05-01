# config.py
from torchvision import transforms
from torchvision.models import VGG19_Weights
CONTENT_DIR = "../datasets/COCO/train2017"
ANNOT_CSV   = "../datasets/Wiki/annotations.csv"
STYLE_ROOT  = "../datasets/Wiki"
# STYLES      = ['barock','abstrakter-expressionismus']
STYLES      = ['barock','abstrakter-expressionismus', 'romantik']
OUT_DIR     = "checkpoints"
SAMPLE_OUT_DIR = "outputs"
GENERATE_OUT_DIR = "outputs/Generator_Output"
IP_ADAPTER_OUT_DIR = "outputs/IP_Adapter_Output"
BATCH_SIZE  = 8
IMG_SIZE    = 512
EPOCHS      = 10
LR          = 2e-4
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT   = 12
TV_WEIGHT      = 1e-7
STYLE_LAYERS   = [0,5,10,19,21]
PREPROCESS = VGG19_Weights.IMAGENET1K_V1.transforms()
# ImageNet normalization stats
IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]

INV_NORMALIZE= transforms.Normalize(
    mean=[-m/s for m,s in zip([0.485,0.456,0.406],[0.229,0.224,0.225])],
    std =[1/s   for s   in           [0.229,0.224,0.225]]
)