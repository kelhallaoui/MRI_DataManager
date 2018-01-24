from DataPreProcessing.utilities import extractNIFTI, readCSV
from DataPreProcessing.ExtractData import extractSlice, resizeImage, transform_to_k_space, plot_k_space
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np


df = readCSV(r'C:/Users/eee/workspace_python/Image Reconstruction/data/ADNI/dataset_metadata.csv', \
			 [0, 3, 4, 6])

#print(df[0:10])
#print(df.dtypes)

for subject_id in df[0:1]['Subject']:
	print('Subject id: ', subject_id)

# Get the T1-weighted MRI image from the datasource and the current subject_id
data, aff, hdr = extractNIFTI(r'C:/Users/eee/workspace_python/Image Reconstruction/data/ADNI/MRI data/', 100206)

print('Data size:', data.shape)

print(aff)

# Extract the slice
axial_slice = 200
img = extractSlice(data, axial_slice)
plt.imshow(img.T, cmap = 'gray')
#plt.show()

img = resizeImage(img, 128)

print(img.shape)

k_img = transform_to_k_space(img)

plot_k_space(k_img)
