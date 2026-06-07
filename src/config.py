from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "GroundTruth.csv"

# Image parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 3
BATCH_SIZE = 32

# Mapping
# Class 0 (Benign): NV, BKL, DF, VASC
# Class 1 (Malignant): MEL, BCC, AKIEC
BENIGN_CLASSES = ['NV', 'BKL', 'DF', 'VASC']
MALIGNANT_CLASSES = ['MEL', 'BCC', 'AKIEC']
