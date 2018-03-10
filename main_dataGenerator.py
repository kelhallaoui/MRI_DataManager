import numpy as np 
import h5py
from DataManager.DataManager import DataManager
from Utilities.utilities import extractNIFTI
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from DataManager.FeatureExtractor import *

dataManager = DataManager(r'C:/Users/eee/workspace_python/Image Reconstruction/data/', ['ADNI'])

#print(dataManager.getData('ADNI', 'Subject'))
#print(dataManager.getKeys('ADNI'))
#coll = dataManager.getDataCollection()

#dataManager.viewSubject('ADNI', 100206)

#print(dataManager.dataCollection['ADNI']['Subject'])

#print(dataManager.data_splits['ADNI'][2].shape)

#dataManager.compileDataset('data', 'ADNI', option = 'image_and_k_space', slice_ix = 0.52, img_shape = 128)

#dataManager.compileDataset('data_gibbs', 'ADNI', option = 'image_and_gibbs', slice_ix = 0.52, img_shape = 128)



dataManager.compileDataset('data_tumor', 'ADNI', option = 'add_tumor', slice_ix = 0.52, img_shape = 128)


data = {}
hf = h5py.File('experiments/data_tumor.h5', 'r')
print([key for key in hf.keys()])
for key in hf.keys():
	print(key)
	vals = list(hf[key])
	data.update({key: np.asarray(vals)})
hf.close()

print(list(data.keys()))

print(data['X_validation'].shape)
print(data['Y_validation'][0])
plt.imshow(np.abs(data['X_validation'][0]).T, cmap = 'gray')
plt.show()


'''
from copy import deepcopy

filepath = r'C:/Users/eee/workspace_python/Image Reconstruction/data/ADNI/MRI data/'
data, aff, hdr = extractNIFTI(filepath, 100206, 'T2')
#data = data[:,:,180]

slice_ix = 180*0.003125
img = extractSlice(data, slice_ix)
img = resizeImage(img, img.shape[0])

img_tumor = add_tumor(deepcopy(img))

phase_map = generate_synthetic_phase_map(img.shape[0])
img = inject_phase_map(img, phase_map)
k_space_img = transform_to_k_space(img)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(np.abs(img/np.max(img)).T, cmap = 'gray')
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(np.abs(k_space_img/np.max(k_space_img)).T, cmap = 'gray')
plt.colorbar()

phase_map = generate_synthetic_phase_map(img_tumor.shape[0])
img_tumor = inject_phase_map(img_tumor, phase_map)
k_space_img = transform_to_k_space(img_tumor)

plt.subplot(2,2,3)
plt.imshow(np.abs(img_tumor/np.max(img_tumor)).T, cmap = 'gray')
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(np.abs(k_space_img/np.max(k_space_img)).T, cmap = 'gray')
plt.colorbar()
plt.show()
'''