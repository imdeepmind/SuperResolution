import os

from logger import logger
from dataset import Dataset

# Constants
DATASET = "data/unlabelled2017"
SAVE_DIR = "data/processed"
GENERATE_DATASET = False

# Checking of the dataset if processed or not
if not os.path.exists(SAVE_DIR + "/high/"):
	GENERATE_DATASET = True

dataset = Dataset()

if GENERATE_DATASET:
	dataset.generate_dataset(DATASET, SAVE_DIR)