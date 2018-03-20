import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from skimage.transform import downscale_local_mean
from pynufft.pynufft import NUFFT_cpu
from copy import deepcopy
import numpy as np
import scipy as scipy
import scipy.ndimage

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
	""" Add a phase map to a real image
	"""
	def polar2z(r,theta):
		return r * np.exp( 1j * theta )

	def z2polar(z):
		return ( np.abs(z), np.angle(z) )

	polar_img = z2polar(img)
	img = polar2z(polar_img[0], phase_map)
	return img

def transform_to_k_space(img, acquisition = 'cartesian', sampling_percent = 1):
	""" Transforms the image to the k-space and shifts the 0-freq to the 
	center.

	Args:
		img (2d numpy array): The image

	Returns 
		A complex 2D matrix with the FFT.
	"""
	if acquisition == 'cartesian':
		freq = np.fft.fft2(img)
		return np.fft.fftshift(freq)
	elif acquisition == 'radial':

		n = img.shape[0]
		total = int(sampling_percent*n*np.pi/2)

		angles = np.repeat(np.arange(0, np.pi, np.pi/total), n)
		radii = np.asarray(list(np.linspace(-1,1,n)) * total)
		om = np.asarray([[r*np.cos(a), r*np.sin(a)] for r, a in zip(radii, angles)])
		om = om * np.pi

		NufftObj = NUFFT_cpu()
		Nd = (n, n)		# image size
		Kd = (2*n, 2*n)	# k-space size
		Jd = (2, 2)  	# interpolation size
		NufftObj.plan(om, Nd, Kd, Jd)

		img = img/np.max(img[:])
		y = NufftObj.forward(img)
		return y.reshape((total, n))

	else:
		raise NameError('Undefined acquisition type! \
			Only \'cartesian\' and \'radial\' implemented')

def introduce_gibbs_artifact(img, percent):
	################ NEED TO CLEAN THIS FUNCTION #############################
	dims = img.shape
	mask = np.ones(dims)

	freq = transform_to_k_space(img)

	percent = percent
	N = np.count_nonzero(mask!=0) - int(round(percent*mask.size))
	np.put(mask, np.random.choice(np.flatnonzero(mask), size=N, replace=False), 0)
	mask[dims[0]//4 : (3*dims[0]//4), dims[1]//4 : (3*dims[1]//4)] = 0

	freq = freq * -1*(mask-1)
	img = np.fft.ifft2(freq)
	return np.abs(img)

def add_tumor(img):
	""" Add a tumor to a 2D image

	A tumor is added at a random location, with a random size, and a 
	random intensity, and Gaussian smoothed.

	Args:
		img (2d numpy array): The image

	Returns:
		img (2d numpy array): The image with the tumor
	"""

	shape = img.shape

	# Random range for the tumor position
	shift_x = np.random.uniform(-1*int(shape[0]//4),int(shape[0]//4))
	shift_y = np.random.uniform(-1*int(shape[1]//4),int(shape[1]//4))

	# Distortion of the x and y axis to get a oval
	dist_x, dist_y= np.random.uniform(0.5,1.5), np.random.uniform(0.5,1.5)

	# Create the matrix within which we will create the tumor
	x = np.arange(-shape[0]//2, shape[0]//2, 1)
	y = np.arange(-shape[1]//2, shape[1]//2, 1)
	xx, yy = np.meshgrid(x, y, sparse=True)

	# The circular tumor
	z = (dist_x*(xx+shift_x))**2 + (dist_y*(yy+shift_y))**2
	rad = 0.05*shape[0]

	# Intensity of the pixels associated with the blood and the tumor
	blood = np.random.uniform(0.8,1)
	tumor = np.random.uniform(0.5,0.8)

	# Size of the blood and tumor regions
	blood_r = np.random.uniform(1.8,2.2)
	tumor_r = np.random.uniform(0.5,1)

	m = np.max(img)
	img[z<int(blood_r*rad**2)] = blood*m
	img[z<int(tumor_r*rad**2)] = tumor*m
	
	# Add a smoothing function to the real and imaginary part 
	img[z<int(4*rad**2)] = scipy.ndimage.filters.gaussian_filter(img[z<int(4*rad**2)], 
															 2, mode='constant')
	return img

def add_gaussian_noise(img, percent):
	################ NEED TO CLEAN THIS FUNCTION #############################
	img_size = img.shape[0]

	r = np.real(img) + percent*np.random.normal(0,1,img_size**2).reshape((img_size,img_size))
	i = np.imag(img) + percent*np.random.normal(0,1,img_size**2).reshape((img_size,img_size))
	return (r + i*1j)

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