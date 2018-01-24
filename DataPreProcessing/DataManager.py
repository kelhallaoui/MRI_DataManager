from utilities import extractNIFTI, readCSV

class DataManager(object):

	def __init__(self, filepath = None):
		self.data_source = filepath
		self.df = None

		if 'ADNI' in self.data_source:
			self.df = readCSV(filepath, [0, 3, 4, 6])

	def getSubjectIDs(self):
		pass

	def setFilepath(self, filepath):
		self.data_source = filepath

dataManager = DataManager(r'C:/Users/eee/workspace_python/Image Reconstruction/data/ADNI/dataset_metadata.csv')
print(dataManager.df)