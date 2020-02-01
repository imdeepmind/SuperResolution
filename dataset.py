import os
import cv2
import numpy as np
from tqdm import tqdm

class Dataset:
	def __init__(self):
		"""
			Constructor method for setting some initial values
		"""
		self.counter = 0
		self.high_images = []

	def read_image(self, path):
		"""
			Method for reading images

			Args:
				path: Path of the image

		"""
		image = cv2.imread(path, 1)

		if image is None:
			raise Exception("Not able to read the image, Path: {}".format(path))

		return image

	def resize_image(self, image, size, interpolation=cv2.INTER_CUBIC):
		"""
			Method for resizing images

			Args:
				image: Image that we want to resize

				size: Size of the target image

				interpolation: Interpolation method for resizing
		"""
		return cv2.resize(image, size, interpolation=interpolation)

	def make_folder(self, folder):
		""" 
			Method for creating folders

			Args:
				folder: Name of the folder (complete path)
		"""

		if not os.path.exists(folder):
			os.makedirs(folder)

	def generate_dataset(self, image_path, save_dir):
		"""
			Mathod for generating the dataset from the raw data

			Args:
				image_path: Path of the dataset

				save_dir: Directory where we want to store the preprocessed images
		"""
		images = os.listdir(image_path)

		if len(images) > 0:
			for image in tqdm(images):
				original = self.read_image(image_path + "/" + image)

				high = self.resize_image(original, (224, 224))
				low = self.resize_image(original, (112, 112))
				high_interpolation = self.resize_image(low, (224, 224))

				self.make_folder(save_dir + "/high/")
				self.make_folder(save_dir + "/low/")

				cv2.imwrite(save_dir + "/high/" + image, high)
				cv2.imwrite(save_dir + "/low/" + image, high_interpolation)

		else:
			raise Exception('There are no images in the folder')

	def get_image(self, image, dir):
		"""
			Method for reading one set of image

			Args:
				image: Name of the image

				dir: Directory where the image is stored
		"""
		high = self.read_image(dir + "/high/" + image)
		low = self.read_image(dir + "/low/" + image)

		return high, low

	def get_images(self, dir, batch_size=32):
		"""
			Method for getting batch of images

			Args:
				dir: Directory where the classes of images are stored

				batch_size: Batch size for the images
		"""
		if len(self.high_images) <= 0:
			self.high_images = os.listdir(dir + "/high/")

		if len(self.high_images) <= 0:
			raise Exception("There are no images for the generator to read")


		while True:
			high_slice = self.high_images[self.counter:self.counter+batch_size]

			self.counter += batch_size

			highs = []
			lows = []

			for image in self.high_images:
				high, low = self.get_image(image, dir)

				highs.append(high / 255.0)
				lows.append(low / 255.0)

			yield np.array(highs), np.array(lows)
