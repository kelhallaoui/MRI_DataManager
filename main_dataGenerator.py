import numpy as np 
import h5py
from DataPreProcessing.utilities import write_data

a = np.random.random(size=(100,20))
b = np.random.random(size=(100,))

write_data(a, b, "wowow.h5")





