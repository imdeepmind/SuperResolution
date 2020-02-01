import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization

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

if GENERATE_DATASET:
	logger.info('Generating the dataset')
	dataset.generate_dataset(DATASET, SAVE_DIR)

model = Sequential()

model.add(Conv2D(filters=128, kernel_size= (9,9), padding= 'same', activation = 'relu', input_shape=(224, 224, 1)))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size= (5, 5), padding= 'same', activation = 'relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=1, kernel_size= (5,5), padding= 'same', activation = 'linear'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

logger.info(model.summary())

