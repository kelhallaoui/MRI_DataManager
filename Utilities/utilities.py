import math
import nibabel as nib
import numpy as np
import csv
import pandas as pd
import zipfile
import gzip
import h5py
import shutil
import skimage
import os
import contextlib
import tempfile
# This package seems to introduce some problems on Windows.
#import medpy.io as mio

def read_CSV(filename, use_cols):
	""" Reads a CSV into a pandas dataframe.

	Args: 
		filename (string): Absolute filename
		use_cols (list of integers): the index of the columns you want to read.

	Returns: 
		df: pandas dataframe
	"""
	df = pd.read_csv(filename, usecols=use_cols)
	return df

def extract_NIFTI(filepath, subject_id, scan_type = 'T1'):
	""" Open a zip file and read a file within it

	Args: 
		zip_filename (string): Absolute filename
		filename (string): filename of the file inside the zip
		scan_type (strng): The acquisition type to extract ('T1' or 'T2')

	Returns: 
		data
	"""
	print('Filepath: ', filepath + str(subject_id) + '_3T_Structural_unproc.zip')
	zip_filename = filepath + str(subject_id) + '_3T_Structural_unproc.zip'
	# If the zip file is not found.
	if not zipfile.is_zipfile(zip_filename):
		raise NameError('Not a valid .zip file.')

	# Open the zip file
	if scan_type == 'T1':
		filename = str(subject_id) + '/unprocessed/3T/T1w_MPR1/' + \
				   str(subject_id) + '_3T_T1w_MPR1.nii.gz'

		with zipfile.ZipFile(zip_filename, 'r') as zf:
			# If the internal file is not found.
			if not filename in zf.namelist():
				raise NameError('Filename not found in the zipfile!')
			file = zf.extract(filename)
			data, aff, hdr = open_NIFTI(file)
			# Delete the file in the root
			shutil.rmtree(str(subject_id) + '/', ignore_errors=True) 
			return data, aff, hdr

	elif scan_type == 'T2':
		filename_t2 = str(subject_id) + '/unprocessed/3T/T2w_SPC1/' + \
					  str(subject_id) + '_3T_T2w_SPC1.nii.gz'
		filename_bc = str(subject_id) + '/unprocessed/3T/T2w_SPC1/' + \
					  str(subject_id) + '_3T_BIAS_BC.nii.gz'
		filename_ch = str(subject_id) + '/unprocessed/3T/T2w_SPC1/' + \
					  str(subject_id) + '_3T_BIAS_32CH.nii.gz'
	
		print('Filename: ', filename_t2)
		with zipfile.ZipFile(zip_filename, 'r') as zf:
			# If the internal file is not found.
			if not filename_t2 in zf.namelist():
				raise NameError('Filename not found in the zipfile!')
	
			# Extract the NIFTI file and read its contents
			####################### Need FIX ###################################
			# File is currently extracted to the root directory 
			# The NIFTI file is then read from this file. 
			#
			# The nib.load(filename) function looks is os.path.exists(filename)
			# This function returns false when .zip is found in the filename
			####################################################################
			file = zf.extract(filename_t2)
			data_t2, aff, hdr = open_NIFTI(filename_t2)
	
			if not filename_bc in zf.namelist():
				return data_t2, aff, hdr
			
			# B1 correction of the T2 images
			file = zf.extract(filename_bc)
			file = zf.extract(filename_ch)
			data_bc, aff, hdr = open_NIFTI(filename_bc)
			data_ch, aff, hdr = open_NIFTI(filename_ch)
	
			data_bc = np.swapaxes(data_bc, 0, 1)
			data_bc = np.swapaxes(data_bc, 0, 2)
			data_bc = np.rot90(np.rot90(data_bc))
	
			data_ch = np.swapaxes(data_ch, 0, 1)
			data_ch = np.swapaxes(data_ch, 0, 2)
			data_ch = np.rot90(np.rot90(data_ch))
	
			data_bc[data_bc==0] = np.finfo(float).eps
			data_ch[data_ch==0] = 1
	
			ratio = data_bc/data_ch
			ratio = skimage.transform.resize(ratio, data_t2.shape, mode='constant')
			t2 = np.multiply(ratio, data_t2)
	
			shutil.rmtree(str(subject_id) + '/', ignore_errors=True) # Delete the file in the root
			return t2, aff, hdr
	else: raise NameError('Invalid acquisition. Either \'T1\' or \'T2\'')


def extract_FigShare(filepath, filenames):
	zip_files = [r'brainTumorDataPublic_1-766.zip',
				 r'brainTumorDataPublic_767-1532.zip',
				 r'brainTumorDataPublic_1533-2298.zip',
				 r'brainTumorDataPublic_2299-3064.zip']

	slices = []
	for filename in filenames:
		if int(filename[:filename.index('.')]) <= 766: file_ix = 0
		elif int(filename[:filename.index('.')]) <= 1532: file_ix = 1
		elif int(filename[:filename.index('.')]) <= 2298: file_ix = 2
		elif int(filename[:filename.index('.')]) <= 3064: file_ix = 3
		di = filepath + zip_files[file_ix] 
		image = open_FigShare_zip(di, filename)
		slices.append(image)
	return slices

def extract_BRATS(filepath, scan_type='T1'):
	keyword = ''
	if scan_type == 'T1':
		keyword = 'MR_T1'
	elif scan_type == 'T2':
		keyword = 'MR_T2'
	filepath_ = ''
	for directory in os.listdir(filepath):
		if os.path.isdir(os.path.join(filepath, directory)):
			if keyword in directory.split('.'):
				for file in os.listdir(os.path.join(filepath, directory)):
					if file.split('.')[-1] == 'mha':
						filepath_ = os.path.join(filepath, directory, file)
						break
				break

	data, header = open_BRATS(filepath_)
	return data, header


def open_NIFTI(filename):
	"""Imports data from the NIFTI files

	Args:
		filename (string): filename and path to the NIFTI file

	Returns:
		data (3D numpy): the 3d volume of the image
		hdr_data: header of the NIFTI file
	"""
	print('FILENAME NIFTI', filename)
	img_mri = nib.load(filename)
	data = img_mri.get_data()
	aff = img_mri.affine
	hdr_data = img_mri.header
	return data, aff, hdr_data

def openFigShare(filepath):
	"""Import data from FigShare files

	Args:
		filepath (string): filepath to the FigShare file

	Returns:
		data (2D numpy): the slice of 2D image of brain with tumor
	"""
	print('FILEPATH FigShare', filepath)
	with h5py.File(filepath, 'r') as f:
		image = np.array(f['cjdata']['image'])
	return image

def open_FigShare_zip(zip_filepath, filename):
	"""Import data from FigShare zip file

	Args:
		zip_filepath (string): filepath to the FigShare zip
		filename (string): filename of the FigShare file in zip to be extracted

	Returns:
		data (2D numpy): the slice of 2D image of brain with tumor
	"""
	print('FILEPATH FigShare in zip {} of name {}'.format(zip_filepath, filename))
	with zipfile.ZipFile(zip_filepath, 'r') as zf:
		with open_h5_in_memory(zf.read(filename)) as f:
			cjdata = f['cjdata']
			image = np.array(cjdata['image'])
	return image

def open_BRATS(filepath):
	"""Import data from FigShare files

	Args:
		filepath (string): filepath to the FigShare file

	Returns:
		data: the 3D image of brain with tumor
	"""
	print('FILEPATH BRATS {}'.format(filepath))
	data, header = mio.load(filepath)
	data = np.array(data).transpose()
	return data, header


def get_FigShare_filemap(directory):
	"""Get the map of {patient_id: slice_file} of FigShare data

	Args:
		directory (string): filepath to the FigShare directory
	Returns:
		map: key -> patient id, value -> array of slice filenames of mri
	"""

	zip_files = [
		'brainTumorDataPublic_1-766.zip',
		'brainTumorDataPublic_767-1532.zip',
		'brainTumorDataPublic_1533-2298.zip',
		'brainTumorDataPublic_2299-3064.zip'
	]

	slices = {}
	for zip_file in zip_files:
		sub_dir_path = os.path.join(directory, zip_file.split('.')[0])
		if os.path.isdir(sub_dir_path):
			for filename in sorted(os.listdir(sub_dir_path), key=lambda name: int(name.split('.')[0])):
				with h5py.File(os.path.join(sub_dir_path, filename), 'r') as f:
					pid = f['cjdata']['PID']
					pid = u''.join(chr(c) for c in pid)
					if pid in slices:
						slices[pid].append(filename)
					else:
						slices[pid] = [filename]
		else:
			with zipfile.ZipFile(os.path.join(directory, zip_file), 'r') as zf:
				for filename in sorted(zf.namelist(), key=lambda name: int(name.split('.')[0])):
					with open_h5_in_memory(zf.read(filename)) as f:
						pid = f['cjdata']['PID']
						pid = u''.join(chr(c) for c in pid)
						if pid in slices:
							slices[pid].append(filename)
						else:
							slices[pid] = [filename]
	
	d = {'Patient ID': [i for i in slices], 'File Map': [slices[i] for i in slices]}
	df = pd.DataFrame(d)
	df = df[['Patient ID', 'File Map']]						
	return df

def get_BRRATS_filemap(directory):
	"""Get the map of {subject_id: mri} of BRATS data

	Args:
		directory (string): fielapth to the BRATS directory
	Returns:
		map: key -> subject id, value -> mri filepath
	"""

	subjects_dirs = []
	for path, subdirs, files in os.walk(directory):
		for file in files:
			filepath = os.path.join(path, file)
			if filepath.split('.')[-1] == 'mha' :
				subjects_dir = '/'.join(filepath.split('/')[0:-3])
				if subjects_dir not in subjects_dirs:
					subjects_dirs.append(subjects_dir)

	subject_filemap = {}
	subject_id = 1
	for subjects_dir in subjects_dirs:
		for subject in os.listdir(subjects_dir):
			filepath = os.path.join(subjects_dir, subject)
			if os.path.isdir(filepath):
				subject_filemap[str(subject_id)] = filepath
				subject_id += 1
			

	d = {'Subject': [i for i in subject_filemap], 'File Map': [subject_filemap[i] for i in subject_filemap]}
	df = pd.DataFrame(d)
	df = df[['Subject', 'File Map']]
	return df

def write_data(databases, attributes, filename):
	""" Writes databases and attributes to a .h5 file

	Args:
		databases (dic): A dictionary of numpy containers which will be stored in .h5 file
		attributes (dic): A dictionary of attributes detailing the parameters used to generate
		                  the data.
		filename (str): the filename of the .h5 file.

	"""
	print('writinng data to experiments/{}.h5 ...'.format(filename))
	if not os.path.exists('experiments/'):
		os.makedirs('experiments/')

	hf = h5py.File('experiments/'+filename+'.h5', "w")

	# Write the attributes to the .h5 dataset
	for attribute in attributes:
		print(attribute, ': ', attributes[attribute])
		hf.attrs[attribute] = attributes[attribute]

	# Write the data to the database
	for database in databases:
		print(database, ': ', databases[database].shape)
		hf.create_dataset(database, data=databases[database])

	hf.close()

def read_data(filename):
	""" Read the data from .h5 file

	Args:
		filename (str): The path to the .h5 file
	"""
	hf = h5py.File(filename, 'r')
	data = {}
	for key in hf.keys():
		vals = list(hf[key])
		data.update({key: np.asarray(vals)})
	hf.close()
	return data

@contextlib.contextmanager
def open_h5_in_memory(hdf5_data):
	file_access_property_list = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
	file_access_property_list.set_fapl_core(backing_store=False)
	file_access_property_list.set_file_image(hdf5_data)

	file_id_args = {
		'fapl': file_access_property_list,
		'flags': h5py.h5f.ACC_RDONLY,
		'name': next(tempfile._get_candidate_names()).encode(),
	}

	h5_file_args = {'backing_store': False, 'driver': 'core', 'mode': 'r'}

	with contextlib.closing(h5py.h5f.open(**file_id_args)) as file_id:
		h5_file = h5py.File(file_id, **h5_file_args)
		try:
			yield h5_file
		finally:
			h5_file.close()