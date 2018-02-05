import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from DataManager.utilities import extractNIFTI, readCSV, write_data
from DataManager.PreProcessData import *
from DataManager.FeatureExtractor import *
import numpy as np
import zipfile

class DataManager(object):
	""" Organization of the data

	The class contains a dictionary with the metadata of all the datasets that we will be using.
	It is a means to organize and aggregate data from different sources. It is also used as a wrapper to
	fetch the data from these different sources and combine them.

	Attrs:
		filepath (string): Filepath to the root folder where the datasets are stored
		                   Requried file structure
		                      - filepath
		                         - ADNI (folder)
		                            - MRI data (folder): contains the NIFTI files
		                            - dataset_metadata.csv: file comtaining metadata
		dataCollection (dictionary): Contains the metadata for all the datasets stored indexed by 
		                             the name of the dataset.
		data_splits (dictionary): Contains the train/validation/test indices for each of the 
		                          datasets in dataCollection.
	"""

	def __init__(self, filepath, datasets = None):
		self.filepath = filepath
		self.dataCollection = None
		self.data_splits = None
		# Add the datasets that are listed to the data collection
		self.addDatasets(datasets)

	def addDatasets(self, datasets):
		""" Add new datasets to the dataCollection dictionary

		Args:
			datasets (list of strings): List of strings with the name of all the datasets
										to add to the collection.
		"""
		for dataset in datasets:
			filepath = self.filepath + str(dataset) + '/dataset_metadata.csv'
			if self.dataCollection is None:
				if dataset is 'ADNI': 
					self.dataCollection = {str(dataset): readCSV(filepath, [0, 3, 4, 6])}
					self.train_validate_test_split(dataset, 'Subject')
			else:
				if dataset is 'ADNI': self.dataCollection[str(dataset)] = readCSV(filepath, [0, 3, 4, 6])

	def train_validate_test_split(self, dataset, column_header, train_percent=.6, validate_percent=.2, seed=None):
		"""Splits up the index associated with the dataset into a train/validation/test set
		
		Args:
			dataset (string): Name of the dataset in question
			column_header (string): Name of the column header with the index

		Returns:
			Adds a new train/validation/test set to the data_splits dictionary.
		"""
		if not seed is None: np.random.seed(seed)
		
		data = self.dataCollection[dataset][column_header]
		perm = np.random.permutation(data)
		m = len(data)

		train_end = int(train_percent * m)
		validate_end = int(validate_percent * m) + train_end

		train = perm[:train_end]
		validate = perm[train_end:validate_end]
		test = perm[validate_end:]
		if self.data_splits is None:
			self.data_splits = {str(dataset): [train, validate, test]}
		else:
			self.data_splits[str(dataset)] = [train, validate, test]
		return

	def compileDataset(self, dataset, option = 'image_and_k_space', slice_ix = 0.52, img_shape = 128):
		""" Extracts the features for the datasets and compiles them into a database

		Args:
			dataset (string): the dataset from which to extract features. 

		"""
		# Extract features 
		featureExtractor = FeatureExtractor(option, slice_ix = slice_ix, img_shape = img_shape, sequence = 3)

		if dataset is 'ADNI': 
			filepath = self.filepath + r'ADNI/MRI data/'
			key = 'Subject'

		subjects = self.dataCollection[dataset][key]
		data_img_space, data_k_space = featureExtractor.extractFeatureSet(subjects[0:2], dataset, filepath)

		databases = {'image_space': data_img_space, 'k_space': data_k_space}
		attributes = {'dataset': dataset, 'slice_ix': slice_ix, 'img_size': img_shape}

		write_data(databases, attributes, 'data.h5')

	def getDataCollection(self):
		return self.dataCollection

	def getData(self, dataset, key):
		if dataset in self.dataCollection:
			if key in self.dataCollection[dataset].columns:
				return self.dataCollection[dataset][key]

	def getKeys(self, dataset):
		if dataset in self.dataCollection:
			return self.dataCollection[dataset].keys()

	def viewSubject(self, dataset, subject_id, slice_ix = 0.5):
		if dataset is 'ADNI':
			# Get the T1-weighted MRI image from the datasource and the current subject_id
			data, aff, hdr = extractNIFTI(self.filepath + r'ADNI/MRI data/', subject_id)
			img = extractSlice(data, slice_ix)
			plt.imshow(img.T, cmap = 'gray')
			plt.colorbar()
			plt.show()
