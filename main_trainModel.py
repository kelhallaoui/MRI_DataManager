from ModelManager.ModelManager import *

modelManager = ModelManager()
modelManager.build_model((28,28))

print(modelManager.getModel().summary())


