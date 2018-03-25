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
	elif scan_type == 'T2':
		filename_t2 = str(subject_id) + '/unprocessed/3T/T2w_SPC1/' + \
		              str(subject_id) + '_3T_T2w_SPC1.nii.gz'
		filename_bc = str(subject_id) + '/unprocessed/3T/T2w_SPC1/' + \
		              str(subject_id) + '_3T_BIAS_BC.nii.gz'
		filename_ch = str(subject_id) + '/unprocessed/3T/T2w_SPC1/' + \
		              str(subject_id) + '_3T_BIAS_32CH.nii.gz'


	else: raise NameError('Invalid acquisition. Either \'T1\' or \'T2\'')
		       
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

def write_data(databases, attributes, filename):
	if not os.path.exists('experiments/'):
		os.makedirs('experiments/')

	hf = h5py.File('experiments/'+filename+'.h5', "w")

	for attribute in attributes:
		print(attribute, ': ', attributes[attribute])
		hf.attrs[attribute] = attributes[attribute]

	for database in databases:
		print(database, ': ', databases[database].shape)
		hf.create_dataset(database, data=databases[database])

	hf.close()

def read_data(filename):
	hf = h5py.File(filename, 'r')
	data = {}
	for key in hf.keys():
		vals = list(hf[key])
		data.update({key: np.asarray(vals)})
	hf.close()
	return data