import cv2
import os
from tqdm import tqdm

from utils import makeFolder

images = os.listdir('data/unlabeled2017')

makeFolder('data/high')
makeFolder('data/low')

for image in tqdm(images):
    original = cv2.imread('data/unlabeled2017/' + image, 1)

    high = cv2.resize(original, (224, 224))
    low = cv2.resize(high, (112, 112))
    high_interpolation = cv2.resize(low, (224, 224), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite('data/high/' + image, high)
    cv2.imwrite('data/low/' + image, high_interpolation)