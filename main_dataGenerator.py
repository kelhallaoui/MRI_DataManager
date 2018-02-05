import numpy as np 
import h5py
from DataManager.utilities import write_data
from DataManager.DataManager import DataManager

dataManager = DataManager(r'C:/Users/eee/workspace_python/Image Reconstruction/data/', ['ADNI'])
#print(dataManager.getData('ADNI', 'Subject'))
#print(dataManager.getKeys('ADNI'))
#coll = dataManager.getDataCollection()

dataManager.viewSubject('ADNI', 100206)

#dataManager.extractFeatures('ADNI', 100206)

#print(dataManager.dataCollection['ADNI']['Subject'])

#print(dataManager.data_splits['ADNI'][2].shape)

#dataManager.compileDataset('ADNI')


#hf = h5py.File('data.h5', 'r')
#print([key for key in hf.keys()])
#data = np.array(hf.get('image_space'))
#hf.close()

#print(data.shape)

#print(data[700, :, :])