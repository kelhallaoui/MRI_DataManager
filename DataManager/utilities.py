import nibabel as nib
import numpy as np
import csv
import pandas as pd
import zipfile
import gzip
import h5py
import shutil

def readCSV(filename, use_cols):
	""" Reads a CSV into a pandas dataframe.

	Args: 
		filename (string): Absolute filename
		use_cols (list of integers): the index of the columns you want to read.

	Returns: 
		df: pandas dataframe
	"""
	df = pd.read_csv(filename, usecols=use_cols)
	return df

def extractNIFTI(filepath, subject_id):
	""" Open a zip file and read a file within it

	Args: 
		zip_filename (string): Absolute filename
		filename (string): filename of the file inside the zip

	Returns: 
		data
	"""
	print('Filepath: ', filepath + str(subject_id) + '_3T_Structural_unproc.zip')
	zip_filename = filepath + str(subject_id) + '_3T_Structural_unproc.zip'
	filename = str(subject_id) + '/unprocessed/3T/T1w_MPR1/' + str(subject_id) + '_3T_T1w_MPR1.nii.gz'
	print('Filename: ', str(subject_id) + '/unprocessed/3T/T1w_MPR1/' + str(subject_id) + '_3T_T1w_MPR1.nii.gz')


	# If the zip file is not found.
	if not zipfile.is_zipfile(zip_filename):
		raise NameError('Not a valid .zip file.')

	# Open the zip file
	with zipfile.ZipFile(zip_filename, 'r') as zf:
		# If the internal file is not found.
		if not filename in zf.namelist():
			raise NameError('Filename not found in the zipfile!')

		# Extract the NIFTI file and read its contents
		####################### Need FIX ###################################
		# File is currently extracted to the root directory 
		# The NIFTI file is then read from this file. 
		#
		# The nib.load(filename) function looks is os.path.exists(filename)
		# This function returns false when .zip is found in the filename
		####################################################################
		file = zf.extract(filename)
		data, aff, hdr = openNIFTI(file)
		shutil.rmtree(str(subject_id) + '/') # Delete the file in the root
		return data, aff, hdr

def openNIFTI(filename):
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

def write_data(X, Y, filename, metadata = None):
	f = h5py.File(filename, "w")
	f.create_dataset('features', data=X)
	f.create_dataset('labels', data=Y)
	f.close()