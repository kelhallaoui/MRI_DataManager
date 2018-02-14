from ExperimentManager.ExperimentManager import *

expManager = ExperimentManager('experiments/data.h5')

print(list(expManager.getData().keys()))

expManager.build_model((128,128), 'autoencoder')

print(expManager.getModel().summary())

expManager.fit_model(5, 1)

