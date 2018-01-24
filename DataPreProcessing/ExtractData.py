import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from skimage.transform import downscale_local_mean
import numpy as np
import scipy as scipy


def resizeImage(img, output_size):
	""" Takes an image and brings it to a specified size
	
	Args:
		img: the input image shape x by y
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

def transform_to_k_space(img):
	""" Transforms the image to the k-space and shifts the 0-freq to the 
	center.

	Args:
		img: A 2D image

	Returns 
		A complex 2D matrix with the FFT.
	"""
	freq = np.fft.fft2(img)
	return np.fft.fftshift(freq)

def plot_k_space(img, setting = 'log'):
	""" Plots the real and imaginary part of an image side by side

	Args:
		img: a 2D image
		setting: None or 'log'
	"""
	plt.subplot(1, 2, 1)
	if setting is 'log': plt.imshow(np.log(np.real(img)), cmap = 'gray_r')
	elif setting is None: plt.imshow(np.real(img), cmap = 'gray_r')
	plt.title('Real Part')

	plt.subplot(1, 2, 2)
	if setting is 'log': plt.imshow(np.log(np.imag(img)), cmap = 'gray_r')
	elif setting is None: plt.imshow(np.imag(img), cmap = 'gray_r')
	plt.title('Imaginary Part')

	plt.show()


def extractSlice(data, slice_ix, orientation = 'axial'):
	if orientation is 'axial':
		return data[:,:,slice_ix]
	elif orientation is 'coronal':
		pass
	elif orientation is 'sagital':
		pass
	else:
		raise NameError('Undefined orientation!')

