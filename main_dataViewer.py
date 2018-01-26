from DataManager.utilities import extractNIFTI, readCSV
from DataManager.PreProcessData import inject_phase_map, generate_synthetic_phase_map, extractSlice, resizeImage, transform_to_k_space, plot_complex_image
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import cmath
import os


df = readCSV(r'C:/Users/eee/workspace_python/Image Reconstruction/data/ADNI/dataset_metadata.csv', \
			 [0, 3, 4, 6])

#print(df[0:10])
#print(df.dtypes)

for subject_id in df[0:4]['Subject']:
	print('Subject id: ', subject_id)

# Get the T1-weighted MRI image from the datasource and the current subject_id
data, aff, hdr = extractNIFTI(r'C:/Users/eee/workspace_python/Image Reconstruction/data/ADNI/MRI data/', 100206)

print('Data size:', data.shape)

# Extract the slice
axial_slice = 0.6
img = extractSlice(data, axial_slice)
img = resizeImage(img, 128)
print('Image')
print('Min', np.amin(img))
print('Max', np.amax(img))

phase_map = generate_synthetic_phase_map(128)
print('Phase map')
print('Min', np.amin(phase_map))
print('Max', np.amax(phase_map))
#plot_complex_image(phase_map.T)

#print(phase_map)

# ix = 1
# for i in range(1,8):
# 	for j in range(1,8):
# 		plt.subplot(7, 7, ix)
# 		plt.imshow(generate_synthetic_phase_map(128), cmap = 'gray')
# 		#plt.colorbar()
# 		ix += 1
# plt.show()

img = inject_phase_map(img, phase_map)

print('Magnitude')
print('Min', np.amin(np.abs(img)))
print('Max', np.amax(np.abs(img)))

print('Angle')
print('Min', np.amin(np.angle(img)))
print('Max', np.amax(np.angle(img)))

plot_complex_image(img.T)
plot_complex_image(transform_to_k_space(img).T, 'polar', 'log')
