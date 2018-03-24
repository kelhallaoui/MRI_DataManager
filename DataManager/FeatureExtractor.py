import numpy as np
import zipfile
from Utilities.utilities import extractNIFTI, extractFigShare
from DataManager.PreProcessData import *

class FeatureExtractor(object):
	""" Extracts features from a dataCollection

	Attrs:
		params (dictionary): contains all the options the user wants for
								 extracting the desired features.
	"""

	def __init__(self, params):
		self.params = params

	def extract_features(self, subjects, dataset, filepath, options=None):
		""" Extracts the desired features for a list of subjects from a dataset

		Args:
			subjects (list): List of the subject IDs
			dataset (string): the name of the dataset 
			filepath (string): path to the data

		Returns:
			A dictionary with all the data matrices
		"""
		if self.params['feature_option'] is 'image_and_k_space':
			data_img_space, data_k_space = self.extract_feature_image_and_k_space(subjects, dataset, filepath, options=options)
			return {'image': data_img_space, 'k_space': data_k_space}

		elif self.params['feature_option'] is 'image_and_gibbs':
			data_img_gibbs, data_gibbs = self.extract_feature_image_and_gibbs(subjects, dataset, filepath, options=options)
			return {'img_with_gibbs': data_img_gibbs, 'gibbs': data_gibbs}

		elif self.params['feature_option'] is 'add_tumor':
			data_img, data_k_space, data_label = self.extract_feature_add_tumor(subjects, dataset, filepath, options=options)
			return {'image': data_img, 'k_space': data_k_space, 'label': data_label}

		elif self.params['feature_option'] is 'denoising':
			data_img, data_img_noisy = self.extract_feature_denoising(subjects, dataset, filepath, options=options)
			return {'image': data_img, 'image_noisy': data_img_noisy}

	
	##########################################################################################
	# 
	# Image and k-space
	#
	##########################################################################################
	def extract_feature_image_and_k_space(self, subjects, dataset, filepath, scan_type = 'T1', options=None):
		if dataset == 'ADNI':
			# Count the number of valid brains in the dataset
			batch_size = 0
			for subject_id in subjects:
				zip_filename = filepath + str(subject_id) + '_3T_Structural_unproc.zip'
				if zipfile.is_zipfile(zip_filename): batch_size += 1
			print('Total subjects: ', batch_size)

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
					data, aff, hdr = extractNIFTI(filepath, subject_id, scan_type)
					for i in range(self.sequence):
						img, k_space_img = self.extract_image_and_k_space(data, slice_ix=self.slice_ix + i*0.003125)
						data_k_space[batch_ix] = k_space_img
						data_img_space[batch_ix] = img
						batch_ix += 1

						print('Subject ID: ', subject_id, '     Slice Index: ', self.slice_ix + i*0.003125)
			return data_img_space, data_k_space
		elif dataset == 'FigShare':
			for subject_id in subjects:
				data_img_space = []
				data_k_space = []
				images = extractFigShare(filepath, subject_id, subject_id_files_map=options['subject_id_files_map'])
				for image in images:
					img_space, k_space = self.extract_image_and_k_space(image)
					data_img_space.append(img_space)
					data_k_space.append(k_space)
			return np.array(data_img_space), np.array(data_k_space, dtype=complex)

	def extract_image_and_k_space(self, data, slice_ix=None):
		""" Extracts the image space and k-space data

		Args:
			data (3D numpy matrix): The volume image
			slice_ix (int): A number between [0, 1] detailing the slice to extract
		
		Returns:
			img: The complex image space
			k_space_img: The complex k-space
		"""
		# Extract the slice if there is slice_ix
		if slice_ix:
			img = extractSlice(data, slice_ix)
		# If not assume the whole data is a slice
		else:
			img = data
		img = resizeImage(img, self.params['img_shape'])
		phase_map = generate_synthetic_phase_map(self.params['img_shape'])
		img = inject_phase_map(img, phase_map)
		k_space_img = transform_to_k_space(img)
		return img, k_space_img

	##########################################################################################
	# 
	# Image with Gibbs and the Gibbs artifact
	#
	##########################################################################################
	def extract_feature_image_and_gibbs(self, subjects, dataset, filepath, scan_type = 'T1', options=None):
		# Count the number of valid brains in the dataset
		batch_size = 0
		for subject_id in subjects:
			zip_filename = filepath + str(subject_id) + '_3T_Structural_unproc.zip'
			if zipfile.is_zipfile(zip_filename): batch_size += 1
		print('Total subjects: ', batch_size)

		# Set containers in which to store the data
		data_img = np.zeros((self.sequence*batch_size, self.img_shape, self.img_shape), dtype=complex)
		data_img_gibbs = np.zeros((self.sequence*batch_size, self.img_shape, self.img_shape), dtype=complex)
		data_gibbs = np.zeros((self.sequence*batch_size, self.img_shape, self.img_shape), dtype=complex)

		# Extract the data
		batch_ix = 0
		for subject_id in subjects:
			zip_filename = filepath + str(subject_id) + '_3T_Structural_unproc.zip'
			if zipfile.is_zipfile(zip_filename):
				# Get the T1-weighted MRI image from the datasource and the current subject_id
				data, aff, hdr = extractNIFTI(filepath, subject_id, scan_type)
				for i in range(self.sequence):
					img, gibbs_img, gibbs = self.extract_image_and_gibbs(data, self.slice_ix + i*0.003125)
					data_img[batch_ix] = img
					data_img_gibbs[batch_ix] = gibbs_img
					data_gibbs[batch_ix] = gibbs
					batch_ix += 1

					print('Subject ID: ', subject_id, '     Slice Index: ', self.slice_ix + i*0.003125)

		return data_img_gibbs, data_gibbs

	def extract_image_and_gibbs(self, data, slice_ix):
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
		gibbs_img = introduce_gibbs_artifact(img, 0.8)
		gibbs = gibbs_img - img
		return img, gibbs_img, gibbs

	##########################################################################################
	# 
	# Image with or without tumor, associated k_space and labels
	#
	##########################################################################################
	def extract_feature_add_tumor(self, subjects, dataset, filepath, options=None):
		# Count the number of valid brains in the dataset
		batch_size = 0
		for subject_id in subjects:
			zip_filename = filepath + str(subject_id) + '_3T_Structural_unproc.zip'
			if zipfile.is_zipfile(zip_filename): batch_size += 1
		print('Total subjects: ', batch_size)

		# Set containers in which to store the data
		data_shape = (self.params['consec_slices']*batch_size, 
					  self.params['img_shape'], 
					  self.params['img_shape'])
		data_img = np.zeros(data_shape, dtype=complex)
		data_label = np.zeros((self.params['consec_slices']*batch_size, 1,), dtype=int)
		data_k_space = []

		# Extract the data
		batch_ix = 0
		for subject_id in subjects:
			zip_filename = filepath + str(subject_id) + '_3T_Structural_unproc.zip'
			if zipfile.is_zipfile(zip_filename):
				# Get the T1-weighted MRI image from the datasource and the current subject_id
				data, aff, hdr = extractNIFTI(filepath, subject_id, self.params['scan_type'])
				for slice_ix in range(self.params['consec_slices']):
					img, k_space_img, label = self.extract_image_add_tumor(data, slice_ix)
					data_img[batch_ix] = img
					data_k_space.append(k_space_img)
					data_label[batch_ix] = label
					batch_ix += 1

					print('Subject ID: ', subject_id, '     Slice Index: ', self.params['slice_ix'] + slice_ix*0.003125)
		return data_img, np.asarray(data_k_space), data_label

	def extract_image_add_tumor(self, data, slice_ix):
		""" Extracts the image space and k-space data

		Args:
			data (3D numpy matrix): The volume image
			slice_ix (int): A number between [0, 1] detailing the slice to extract
		
		Returns:
			img: The complex image space
			k_space_img: The complex k-space
		"""
		# Extract the slice
		img = extractSlice(data, self.params['slice_ix'] + slice_ix*0.003125)
		img = resizeImage(img, self.params['img_shape'])

		label = np.random.randint(0,2)
		if label == 1:
			img = add_tumor(img)

		#phase_map = generate_synthetic_phase_map(self.img_shape)
		#img = inject_phase_map(img, phase_map)
		k_space_img = transform_to_k_space(img, acquisition = self.params['acquisition_option'], 
												sampling_percent = self.params['sampling_percent'])
		return img, k_space_img, label

	##########################################################################################
	# 
	# Image with and without added noise
	#
	##########################################################################################
	def extract_feature_denoising(self, subjects, dataset, filepath, scan_type = 'T2', options=None):
		# Count the number of valid brains in the dataset
		batch_size = 0
		for subject_id in subjects:
			zip_filename = filepath + str(subject_id) + '_3T_Structural_unproc.zip'
			if zipfile.is_zipfile(zip_filename): batch_size += 1
		print('Total subjects: ', batch_size)

		# Set containers in which to store the data
		data_shape = (self.sequence*batch_size, self.img_shape, self.img_shape)
		data_img = np.zeros(data_shape, dtype=complex)
		data_img_noisy = np.zeros(data_shape, dtype=complex)

		# Extract the data
		batch_ix = 0
		for subject_id in subjects:
			zip_filename = filepath + str(subject_id) + '_3T_Structural_unproc.zip'
			if zipfile.is_zipfile(zip_filename):
				# Get the T1-weighted MRI image from the datasource and the current subject_id
				data, aff, hdr = extractNIFTI(filepath, subject_id, scan_type)
				for i in range(self.sequence):
					img, img_noisy = self.extract_image_noise(data, self.slice_ix + i*0.003125)

					data_img[batch_ix] = img
					data_img_noisy[batch_ix] = img_noisy
					batch_ix += 1

					print('Subject ID: ', subject_id, '     Slice Index: ', self.slice_ix + i*0.003125)
		return data_img, data_img_noisy

	def extract_image_noise(self, data, slice_ix):
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
		k_space_img = transform_to_k_space(img)
		k_space_img_noisy = add_gaussian_noise(k_space_img, 0.05)
		return k_space_img, k_space_img_noisy