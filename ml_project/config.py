"""
Configuration for the EM S-parameter ML Classification Project
"""
import os

# Paths
DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
IMAGE_DIR   = os.path.join(OUTPUT_DIR, "images")
MODEL_DIR   = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Data split — with only 42 samples we use LeaveOneOut / small test fraction
TEST_SIZE    = 0.2          # ~8 samples for test
RANDOM_STATE = 42

# EfficientNetV2 image settings
IMG_SIZE      = (96, 96)    # smaller images = much faster on CPU
BATCH_SIZE    = 16          # larger batch = fewer steps
EPOCHS        = 5           # fast training — backbone is frozen (pretrained)
LEARNING_RATE = 1e-3
DROPOUT_RATE  = 0.3

# LightGBM — tuned for small dataset
LGBM_PARAMS = {
    "n_estimators"   : 500,
    "learning_rate"  : 0.03,
    "max_depth"      : 4,
    "num_leaves"     : 15,
    "min_child_samples": 1,   # allow leaves with 1 sample (tiny dataset)
    "subsample"      : 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha"      : 0.1,
    "reg_lambda"     : 0.1,
    "random_state"   : RANDOM_STATE,
    "n_jobs"         : -1,
    "verbose"        : -1,
    "class_weight"   : "balanced",
}
