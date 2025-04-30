# config.py
CONTENT_DIR = "../datasets/COCO/train2017"
ANNOT_CSV   = "../datasets/Wiki/annotations.csv"
STYLE_ROOT  = "../datasets/Wiki"
STYLES      = ['barock','abstrakter-expressionismus']
OUT_DIR     = "checkpoints"
BATCH_SIZE  = 8
IMG_SIZE    = 256
EPOCHS      = 2
LR          = 1e-4
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT   = 10.0
TV_WEIGHT      = 1e-6
STYLE_LAYERS   = [0,5,10,19,21]