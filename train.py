import os

from logger import logger
from dataset import Dataset

# Constants
DATASET = "dataset/unlabeled2017"
SAVE_DIR = "dataset/processed"
GENERATE_DATASET = False

# Checking of the dataset if processed or not
if not os.path.exists(SAVE_DIR + "/high/"):
	GENERATE_DATASET = True

dataset = Dataset()

print(GENERATE_DATASET)

if GENERATE_DATASET:
	dataset.generate_dataset(DATASET, SAVE_DIR)