import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from skimage.transform import downscale_local_mean
import numpy as np
import scipy as scipy

def extractSlice(data, slice_ix, orientation = 'axial'):
	""" Extract a slice from a volumetric image
	
	Args: 
		data (3D numpy array): The volumetric image
		slice_ix (int): The slice to extract
		orientation (string): The orientation to extract
			'axial', 'coronal', 'sagittal'
	"""
	if slice_ix > 1 or slice_ix < 0: 
		raise NameError('The extracted slice should be within [0, 1]. A proportion of the volume size.')
		return None

	if orientation is 'axial':
		slice_ix = int(slice_ix * len(data[0,0,:]))
		return data[:,:,slice_ix]
	elif orientation is 'coronal':
		pass
	elif orientation is 'sagittal':
		pass
	else:
		raise NameError('Undefined orientation!')

def resizeImage(img, output_size):
	""" Takes an image and brings it to a specified size
	
	Args:
		img (2d numpy array): The image
		output_size: the desired output shape
	
	Returns:
		The downscaled image
	"""
	x, y = img.shape

	size = output_size * (max(x,y)//output_size)
	if output_size >= x and output_size >= y:
		size = output_size
	pad_img = scipy.misc.imresize(img, (size, size))
	fx, fy = size//output_size, size//output_size
	return downscale_local_mean(pad_img, (fx, fy))

def generate_synthetic_phase_map(kernel_size = 128, setting = 'sinusoid'):
	""" Generate a synthetic phase map

	Args:
		kernel_size (int): The size of the output phase map
		setting (string): Function used to generate the phase map {'sinusoid'}
	
	Return
		2D matrix containing a synthetic phase map range = [-pi, pi]
	"""
	if setting is 'sinusoid':
		ix_x, ix_y = ((np.indices((2*kernel_size, 2*kernel_size)) - kernel_size) / (kernel_size)) 
		freq_x, freq_y = 2 * np.random.random((1,)), 2 * np.random.random((1,))

		phase_map = np.pi * np.sin( 2*np.pi* (freq_x*ix_x)) * np.sin(2*np.pi*  (freq_y*ix_y) ) 
		
		# Rotate the phase map by a random angle, this will reshape the image
		# We thus extract the center of the image
		rot_angle = 360 * np.random.random((1,))
		phase_map = scipy.ndimage.interpolation.rotate(phase_map, rot_angle, reshape = True)
		ix = phase_map.shape[0]//2 - kernel_size//2
		phase_map = phase_map[ix:ix+kernel_size, ix:ix+kernel_size]
		return phase_map

def inject_phase_map(img, phase_map):
	def polar2z(r,theta):
		return r * np.exp( 1j * theta )

	def z2polar(z):
		return ( np.abs(z), np.angle(z) )

	polar_img = z2polar(img)
	img = polar2z(polar_img[0], phase_map)
	return img

def transform_to_k_space(img):
	""" Transforms the image to the k-space and shifts the 0-freq to the 
	center.

	Args:
		img (2d numpy array): The image

	Returns 
		A complex 2D matrix with the FFT.
	"""
	freq = np.fft.fft2(img)
	return np.fft.fftshift(freq)

def plot_complex_image(img, mode = 'polar', setting = None):
	""" Plots the real and imaginary part of an image side by side

	Args:
		img (2d numpy array): The image
		setting: None or 'log'
	"""
	if mode is 'polar': a, b = np.abs(img), np.angle(img)
	elif mode is 'cartesian': a, b = np.real(img), np.imag(img)

	plt.subplot(1, 2, 1)
	if setting is 'log': 
		a[a == 0] = 1
		plt.imshow(np.log(a), cmap = 'gray')
	elif setting is None: plt.imshow(a, cmap = 'gray')

	if mode is 'polar': plt.title('Absolute Value')
	elif mode is 'cartesian': plt.title('Real Part')

	plt.colorbar()

	plt.subplot(1, 2, 2)
	if setting is 'log': 
		b[b == 0] = 1
		plt.imshow(np.log(b), cmap = 'gray')
	elif setting is None: plt.imshow(b, cmap = 'gray')
	
	if mode is 'polar': plt.title('Phase Value')
	elif mode is 'cartesian': plt.title('Imaginary Part')

	plt.colorbar()
	plt.show()