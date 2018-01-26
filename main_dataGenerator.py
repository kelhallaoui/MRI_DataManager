import numpy as np 
import h5py
from DataManager.utilities import write_data
from DataManager.DataManager import DataManager

dataManager = DataManager(r'C:/Users/eee/workspace_python/Image Reconstruction/data/', ['ADNI'])

coll = dataManager.getDataCollection()

print(dataManager.getData('ADNI', 'Subject'))

dataManager.viewSubject('ADNI', 100206, 0.6)



