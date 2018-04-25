import numpy as np
import zipfile
from Utilities.utilities import extract_NIFTI, extract_FigShare, extract_BRATS
from DataManager.PreProcessData import *

class FeatureExtractor(object):
	""" Extracts features from a dataCollection

	Attrs:
		params (dictionary): contains all the options the user wants for
								 extracting the desired features.
	"""
	def __init__(self, params):
		self.batch_size = 32
		self.params = params

	def extract_features(self, subjects, dataset, filepath, metadata=None):
		""" Extracts the desired features for a list of subjects from a dataset

		Args:
			subjects (list): List of the subject IDs
			dataset (string): the name of the dataset 
			filepath (string): path to the data

		Returns:
			A dictionary with all the data matrices
		"""
		f = self.extract(subjects, dataset, filepath, metadata=metadata)
		for data_batch in f:
			if self.params['feature_option'] is 'add_tumor':
				yield {'image': data_batch[0], 'k_space': data_batch[1], 'label': data_batch[2]}
			elif self.params['feature_option'] is 'image_and_k_space':
				yield {'image': data_batch[0], 'k_space': data_batch[1]}
			elif self.params['feature_option'] is 'image_and_gibbs':
				yield {'img_with_gibbs': data_batch[0], 'gibbs': data_batch[1]}
			elif self.params['feature_option'] is 'denoising':
				yield {'image': data_batch[0], 'image_noisy': data_batch[1]}


	def extract(self, subjects, dataset, filepath, metadata=None):
		# Set containers for the data
		data_shape = (self.batch_size, self.params['img_shape'], self.params['img_shape'])
		if self.params['feature_option'] is 'add_tumor':
			data_img     = np.zeros(data_shape, dtype=complex)
			data_k_space = np.zeros(data_shape, dtype=complex)
			data_label   = np.zeros((self.batch_size, 1,), dtype=int)
			#data_slice_ix = ...
			data_batch = [data_img, data_k_space, data_label]
		elif self.params['feature_option'] is 'image_and_k_space':
			data_k_space   = np.zeros(data_shape, dtype=complex)
			data_img_space = np.zeros(data_shape, dtype=complex)
			data_batch = [data_k_space, data_img_space]
		elif self.params['feature_option'] is 'image_and_gibbs':
			data_img       = np.zeros(data_shape, dtype=complex)
			data_img_gibbs = np.zeros(data_shape, dtype=complex)
			data_gibbs     = np.zeros(data_shape, dtype=complex)
			data_batch = [data_img, data_img_gibbs, data_gibbs]
		elif self.params['feature_option'] is 'denoising':
			data_img       = np.zeros(data_shape, dtype=complex)
			data_img_noisy = np.zeros(data_shape, dtype=complex)
			data_batch = [data_img, data_img_noisy]

		# Extract the data
		batch_ix = 0
		for subject_id in subjects:
			zip_filename = filepath + str(subject_id) + '_3T_Structural_unproc.zip'
			if zipfile.is_zipfile(zip_filename):

				if dataset == 'ADNI':
					data, aff, hdr = extract_NIFTI(filepath, subject_id, self.params['scan_type'])
				elif dataset == 'FigShare':
					raise NameError('Feature Extraction not implemented for FigShare data!')
				elif dataset == 'BRATS':
					raise NameError('Feature Extraction not implemented for BRATS data!')

				for slice_ix in range(self.params['consec_slices']):
					if self.params['feature_option'] is 'add_tumor':
						outputs = self.extract_image_add_tumor(data, slice_ix)
					elif self.params['feature_option'] is 'image_and_k_space':
						outputs = self.extract_image_and_k_space(data, slice_ix)
					elif self.params['feature_option'] is 'image_and_gibbs':
						outputs = self.extract_image_and_gibbs(data, slice_ix)
					elif self.params['feature_option'] is 'denoising':
						outputs = self.extract_image_noise(data, slice_ix)

					for ix, d in enumerate(outputs): data_batch[ix][batch_ix] = d

					print('Subject ID: ', subject_id, '     Slice Index: ', self.params['slice_ix'] + slice_ix*0.003125)
					batch_ix += 1
					if batch_ix%self.batch_size==0:
						batch_ix = 0
						yield data_batch

		# If there are less than batch_size amount of data points remaining return them
		for ix in range(len(data_batch)): data_batch[ix] = data_batch[ix][0:batch_ix]
		yield data_batch

	##########################################################################################
	# 
	# Image with or without tumor, associated k_space and labels
	#
	##########################################################################################
	def extract_image_add_tumor(self, data, slice_ix):
		""" Extracts the image space and k-space data

		Args:
			data (3D numpy matrix): The volume image
			slice_ix (int): A number between [0, 1] detailing the slice to extract
		
		Returns:
			img: The complex image space
			k_space_img: The complex k-space
		"""
		# Get the tumor intensity to be close to the CSF
		tumor_intensity = 255*get_csf_intensity(data)/np.max(np.abs(data))
		# Extract the slice
		img = extract_slice(data, self.params['slice_ix'] + slice_ix*0.003125)
		img = resize_image(img, self.params['img_shape'])
		# Randomly add tumor
		label = np.random.randint(0,2)
		if label == 1: img = add_tumor(img, tumor_intensity,
									   self.params['tumor_option'],
									   self.params['tumor_radius'],
									   self.params['tumor_radius_range'])
		# Transform to k-space
		k_space_img = transform_to_k_space(img, acquisition = self.params['acquisition_option'], 
												sampling_percent = self.params['sampling_percent'])
		return [img, k_space_img, label]

	##########################################################################################
	# 
	# Image and k-space
	#
	##########################################################################################
	def extract_image_and_k_space(self, data, slice_ix = None):
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
			img = extract_slice(data, slice_ix)
		# If not assume the whole data is a slice
		else:
			img = data
		img = resize_image(img, self.params['img_shape'])
		phase_map = generate_synthetic_phase_map(self.params['img_shape'])
		img = inject_phase_map(img, phase_map)
		k_space_img = transform_to_k_space(img)
		return img, k_space_img

	##########################################################################################
	# 
	# Image with Gibbs and the Gibbs artifact
	#
	##########################################################################################
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
		img = extract_slice(data, slice_ix)
		img = resize_image(img, self.img_shape)
		gibbs_img = introduce_gibbs_artifact(img, 0.8)
		gibbs = gibbs_img - img
		return img, gibbs_img, gibbs

	##########################################################################################
	# 
	# Image with and without added noise
	#
	##########################################################################################
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
		img = extract_slice(data, slice_ix)
		img = resize_image(img, self.img_shape)
		k_space_img = transform_to_k_space(img)
		k_space_img_noisy = add_gaussian_noise(k_space_img, 0.05)
		return k_space_img, k_space_img_noisy