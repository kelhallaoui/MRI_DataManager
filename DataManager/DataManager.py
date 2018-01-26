import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from DataManager.utilities import extractNIFTI, readCSV
from DataManager.PreProcessData import extractSlice
import numpy as np

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
	"""

	def __init__(self, filepath, datasets = None):
		self.filepath = filepath
		self.dataCollection = None
		# Add the datasets that are listed to the data collection
		self.addDatasets(datasets)

		# The data seperated into three sets
		#self.train = None
		#self.validate = None
		#self.test = None


	def addDatasets(self, datasets):
		for dataset in datasets:
			filepath = self.filepath + str(dataset) + '/dataset_metadata.csv'
			if self.dataCollection is None:
				if dataset is 'ADNI': self.dataCollection = {str(dataset): readCSV(filepath, [0, 3, 4, 6])}
			else:
				if dataset is 'ADNI': self.dataCollection[str(dataset)] = readCSV(filepath, [0, 3, 4, 6])

	def train_validate_test_split(self, train_percent=.6, validate_percent=.2, seed=None):
		"""
		NEEDS TO BE REWORKED ENTIRELY!!!
		THE TRAINING AND TESTING SETS SHOULD MIMIC THE DATA COLLECTION DICTIONARY!!!
		"""
		if not seed is None: np.random.seed(seed)
		if self.df is None: 
			raise NameError('You must initiate a dataset in order to split!')
			return None
		perm = np.random.permutation(self.df.index)
		m = len(self.df)
		train_end = int(train_percent * m)
		validate_end = int(validate_percent * m) + train_end
		self.train = self.df.ix[perm[:train_end]]
		self.validate = self.df.ix[perm[train_end:validate_end]]
		self.test = self.df.ix[perm[validate_end:]]
		return self.train, self.validate, self.test

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

