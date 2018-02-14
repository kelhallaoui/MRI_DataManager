import numpy as np
import zipfile
from Utilities.utilities import extractNIFTI
from DataManager.PreProcessData import *

class FeatureExtractor(object):
	""" Extracts features from a dataCollection

	Attrs
		option (string): what features to extract
		slice_ix (int): what slice to extract
		img_shape (int): size of the images
		sequence (int): how many subsequent slices to extract
	"""

	def __init__(self, option = 'image_and_k_space', slice_ix = 0.52, img_shape = 128, sequence = 1):
		self.option = option
		self.slice_ix = slice_ix
		self.img_shape = img_shape
		self.sequence = sequence

	def extractFeatureSet(self, subjects, dataset, filepath):
		if self.option is 'image_and_k_space':
			data_img_space, data_k_space = self.extractFeature_image_and_k_space(subjects, dataset, filepath)
			return data_img_space, data_k_space


	def extractFeature_image_and_k_space(self, subjects, dataset, filepath):
		# Count the number of valid brains in the dataset
		batch_size = 0
		for subject_id in subjects:
			zip_filename = filepath + str(subject_id) + '_3T_Structural_unproc.zip'
			if zipfile.is_zipfile(zip_filename): batch_size += 1
		print('Total subjects: ', batch_size)

		# Set containers in which to store the data
		data_k_space = np.zeros((self.sequence*batch_size, self.img_shape, self.img_shape), dtype=complex)
		data_img_space = np.zeros((self.sequence*batch_size, self.img_shape, self.img_shape), dtype=complex)

		# Extract the data
		batch_ix = 0
		for subject_id in subjects:
			zip_filename = filepath + str(subject_id) + '_3T_Structural_unproc.zip'
			if zipfile.is_zipfile(zip_filename):
				# Get the T1-weighted MRI image from the datasource and the current subject_id
				data, aff, hdr = extractNIFTI(filepath, subject_id)
				for i in range(self.sequence):
					img, k_space_img = self.extract_image_and_k_space(data, self.slice_ix + i*0.003125)
					data_k_space[batch_ix] = k_space_img
					data_img_space[batch_ix] = img
					batch_ix += 1

					print('Subject ID: ', subject_id, '     Slice Index: ', self.slice_ix + i*0.003125)

		return data_img_space, data_k_space




	def extract_image_and_k_space(self, data, slice_ix):
		""" Extracts the image space and k-space data

		Args:
			data (3D numpy matrix): The volume image
			slice_ix (int): A number between [0, 1] detailing the slice to extract
		
		Returns:
			img: The complex image space
			k_space_img: The complex k-space
		"""
		# Extract the slice
		img = extractSlice(data, slice_ix)
		img = resizeImage(img, self.img_shape)
		phase_map = generate_synthetic_phase_map(self.img_shape)
		img = inject_phase_map(img, phase_map)
		k_space_img = transform_to_k_space(img)
		return img, k_space_img