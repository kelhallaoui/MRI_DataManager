# Packages which can be used to extract features from MRI data.

From a set of datasets this package can extract features and load them into .h5 files ready for subsequent use with deep learning frameworks. The purpose of this code is to unify the feature extraction and join together similar method which are commonly needed for extracting these features.

## Datasets Supported
- ADNI
- FigShare

## Feature Extraction Options
- Image and k-space
- Image and k-space with added tumors (circular or ring) and corresponding labels
- Image with and without Gibb's artifacts
- Image with and without added Gaussian noise in the real and imaginary part of the k-space



# How to use

An example of how to use this code is provided in **main_dataGenerator.py**. First initialize the DataManager object with the location of the data and the datasets you want to extract data from.

```
dataManager = DataManager(r'C:/Users/.../data/', ['ADNI'])
```

Then detail the types of features you wish to extract and other required system parameters, then extract these features! For example,

```
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

dataManager.compile_dataset(params)
```

---

# Install MedPy for python3
```
sudo pip install nibabel pydicom
sudo pip install https://github.com/loli/medpy/archive/Release_0.3.0p3.zip
```
