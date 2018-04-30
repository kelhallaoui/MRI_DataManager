import numpy as np 
import h5py
from DataManager.DataManager import DataManager
from Utilities.utilities import extract_NIFTI, extract_FigShare
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from DataManager.FeatureExtractor import *

'''
#Example to extract FigShare Dataset
dataManager = DataManager(r'C:/Users/eee/workspace_python/Image Reconstruction/data/', ['FigShare'])
params = {'database_name': 		'fig_share_data',
		  'dataset': 			'FigShare',
		  'feature_option':		'image_and_k_space',
		  'img_shape': 			128,
		  'num_subjects': 		'all'}

print(len(dataManager.dataCollection['FigShare']))
print(len(dataManager.data_splits['FigShare'][0]))
print(len(dataManager.data_splits['FigShare'][1]))
print(len(dataManager.data_splits['FigShare'][2]))

dataManager.compile_dataset(params)
'''

# Example to extract data from the ADNI dataset
dataManager = DataManager(r'C:/Users/eee/workspace_python/Image Reconstruction/data/', ['ADNI'])

params = {'database_name': 		'data_tumor_size5_large',
		  'dataset': 			'ADNI',
		  'batch_size':         32,
		  'feature_option':		'add_tumor',
		  'slice_ix': 			0.52, #0.32, #0.52,
		  'img_shape': 			128,
		  'consec_slices':		120, #120,#30,
		  'num_subjects': 		'all',
		  'scan_type': 			'T2',
		  'acquisition_option':	'cartesian',
		  'sampling_percent': 	1, #0.0625,
		  'accel_factor':       0, # How to implement this?
		  'tumor_option':		'circle',
		  'tumor_diameter':      0.05,
		  'tumor_diameter_range':[0.8,1.2]}

#dataManager.compile_dataset(params)

with h5py.File('experiments/data_tumor_size5_large.h5', 'r') as hf:
	keys = list(hf.keys())
	print(keys)

	dataset = 'validation'
	X_identifier = 'k_space'
	Y_identifier = 'label'

	temp = [i for i in keys if dataset+'_'+X_identifier in i]
	x_dims = hf[temp[0]].shape[1::]
	num_files = len(temp)
	print(num_files)

	# Get dimensions of the output space
	temp = [i for i in keys if dataset+'_'+Y_identifier in i]
	y_dims = hf[temp[0]].shape[1::]

	# Get the number of records per file
	num_records = hf[temp[0]].shape[0]
	total_records = np.sum([hf[i].shape[0] for i in temp])

	print(num_records)
	print(total_records)
	print([hf[i].shape[0] for i in temp])


data = {}
hf = h5py.File('experiments/data_tumor_size5_large.h5', 'r')
print([key for key in hf.keys()])
print('--------------DATA KEYS------------------')
for key in hf.keys():
	print(key)
	#vals = list(hf[key])
	#data.update({key: np.asarray(vals)})
print('------------------------------------------')
print('--------------PARAMETERS------------------')
for item in hf.attrs.keys():
	print(item + ":", hf.attrs[item])
	if item == 'subjects_train':
		print(len(hf.attrs[item]))
print('------------------------------------------')
for key in hf.keys():
	print(key)
	vals = list(hf[key])
	data.update({key: np.asarray(vals)})
	break
hf.close()

print(list(data.keys()))
for d in list(data.keys()):
	print(d, data[d].shape)
ix = 9
plt.imshow(np.abs(np.rot90(data[key][ix])), cmap = 'gray')
plt.show()


'''
plt.subplot(2,2,1)
plt.imshow(np.abs(np.rot90(data['train_image'][ix])), cmap = 'gray')
plt.subplot(2,2,2)
plt.imshow(np.abs(np.rot90(data['train_image'][(ix + params['consec_slices'] - 1)//3])), cmap = 'gray')
plt.subplot(2,2,3)
plt.imshow(np.abs(np.rot90(data['train_image'][2*(ix + params['consec_slices'] - 1)//3])), cmap = 'gray')
plt.subplot(2,2,4)
plt.imshow(np.abs(np.rot90(data['train_image'][ix + params['consec_slices'] - 1])), cmap = 'gray')
plt.show()
'''


'''
from copy import deepcopy

filepath = r'C:/Users/eee/workspace_python/Image Reconstruction/data/ADNI/MRI data/'
data, aff, hdr = extract_NIFTI(filepath, 100206, 'T2')
#data = data[:,:,180]

slice_ix = 180*0.003125
img = extract_slice(data, slice_ix)
img = resize_image(img, img.shape[0])

img_tumor = add_tumor(deepcopy(img))

phase_map = generate_synthetic_phase_map(img.shape[0])
img = inject_phase_map(img, phase_map)
k_space_img = transform_to_k_space(img)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(np.abs(img/np.max(img)).T, cmap = 'gray')
plt.axis('off')
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(np.abs(k_space_img/np.max(k_space_img)).T, cmap = 'gray')
plt.axis('off')
plt.colorbar()

phase_map = generate_synthetic_phase_map(img_tumor.shape[0])
img_tumor = inject_phase_map(img_tumor, phase_map)
k_space_img = transform_to_k_space(img_tumor)

plt.subplot(2,2,3)
plt.imshow(np.abs(img_tumor/np.max(img_tumor)).T, cmap = 'gray')
plt.axis('off')
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(np.abs(k_space_img/np.max(k_space_img)).T, cmap = 'gray')
plt.axis('off')
plt.colorbar()
plt.show()
'''