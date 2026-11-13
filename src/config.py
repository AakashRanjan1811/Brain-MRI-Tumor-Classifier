"""Project-wide constants."""

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DISPLAY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary",
}
NUM_CLASSES = len(CLASSES)
SEED = 42
VAL_FRACTION = 0.15
KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
