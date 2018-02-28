from ExperimentManager.ExperimentManager import *

expManager = ExperimentManager('experiments/data.h5')

print(list(expManager.getData().keys()))

expManager.build_model((64,64), 'autoencoder')

print(expManager.getModel().summary())

expManager.fit_model(5, 128)

