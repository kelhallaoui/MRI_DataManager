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

# Install MedPy for python3
```
sudo pip install nibabel pydicom
sudo pip install https://github.com/loli/medpy/archive/Release_0.3.0p3.zip
```
