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

An example of how to use this code is provided in **main_dataGenerator.py**. First initialize the DataManager object with the location of the data and the datasets you want to extract data from. Then detail the types of features you wish to extract and other required system parameters, then extract these features! 

For example, to extract k-space from the ADNI dataset with simulated tumors you can use the following bit of code. This will take 120 slices from each subject in the ADNI dataset and inject a 5% field of view (FOV) diameter tumor in 50% of the instances. The data will be split into a training, validation and testing set, each of these subsets will be binned into batches to be fed to a machine learning model. 

```
dataManager = DataManager(r'C:/Users/.../data/', ['ADNI'])

params = {'database_name':       'data_tumor_size5',
          'dataset':             'ADNI',
          'batch_size':          32,
          'feature_option':      'add_tumor',
          'slice_ix':            0.52,
          'img_shape':           128,
          'consec_slices':       120,
          'num_subjects':        'all',
          'scan_type':           'T2',
          'acquisition_option':  'cartesian',
          'sampling_percent':    1, 
          'accel_factor':        0, 
          'tumor_option':        'circle',
          'tumor_diameter':      0.05,
          'tumor_diameter_range':[0.8,1.2]}

dataManager.compile_dataset(params)
```

The resulting file is a .h5 database that is stored in the experiments folder. The extracted features are stored in batches with keys formated as **subset_identifier_ix**, subset is {train, validation, test}, the identifier is {k_space, image, label} (this will differ based on the feature_option) and ix is the index of the batch.

---
# Acknowledgements

We ask that any works which use this package for feature extraction acknowledge the authors of MRI DataManager Karim El Hallaoui and Anson Leung.

---

# Install MedPy for python3
```
sudo pip install nibabel pydicom
sudo pip install https://github.com/loli/medpy/archive/Release_0.3.0p3.zip
```
