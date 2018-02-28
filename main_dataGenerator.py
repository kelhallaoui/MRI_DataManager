import numpy as np 
import h5py
from DataManager.DataManager import DataManager

dataManager = DataManager(r'C:/Users/eee/workspace_python/Image Reconstruction/data/', ['ADNI'])
#print(dataManager.getData('ADNI', 'Subject'))
#print(dataManager.getKeys('ADNI'))
#coll = dataManager.getDataCollection()

#dataManager.viewSubject('ADNI', 100206)

#print(dataManager.dataCollection['ADNI']['Subject'])

#print(dataManager.data_splits['ADNI'][2].shape)

#dataManager.compileDataset('data_ext', 'ADNI', option = 'image_and_k_space', slice_ix = 0.52, img_shape = 64)

dataManager.compileDataset('data_gibbs', 'ADNI', option = 'image_and_gibbs', slice_ix = 0.52, img_shape = 128)


data = {}
hf = h5py.File('experiments/data_ext.h5', 'r')
print([key for key in hf.keys()])
for key in hf.keys():
	print(key)
	vals = list(hf[key])
	data.update({key: np.asarray(vals)})
hf.close()

print(list(data.keys()))

print(data['X_validation'].shape)
