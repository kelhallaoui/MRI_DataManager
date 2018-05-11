from ExperimentManager.ModelZoo import *
from Utilities.utilities import read_data
import numpy as np
import os

class ExperimentManager(object):
	""" Sets up the machine learning experiment

	Builds a model and organizes a file which will store the model and the trained weights

	Attrs:
		exp_name (string): the name of the current experiment
		model_name (string): this determines the model which will be loaded in the zoo
		input_shape (tuple): size of the input tensors
		model (object): The model object
	"""

	def __init__(self, data_filename, exp_name = 'testing'):
		self.data_filename = data_filename
		self.exp_name = exp_name
		self.model_name = None
		self.input_shape = None
		self.model = None
		self. data = self.load_data(data_filename)

	def build_model(self, input_shape, model_name = 'autoencoder'):
		self.model_name = model_name
		self.input_shape = input_shape
		self.model = buildModel(model_name, input_shape)

	def fit_model(self, epochs, batch_size):
		if self.model is None: raise NameError('Deep learning architecture not set!')

		# Save the model
		self.save_model()

		x_train = self.data['X_train']
		y_train = self.data['Y_train']
		x_test = self.data['X_validation']
		y_test = self.data['Y_validation']

		print(x_train.shape)

		# Set up channels last for the data be passed through the model
		x_train = np.reshape(x_train, x_train.shape + (1,))
		y_train = np.reshape(y_train, y_train.shape + (1,))
		x_test  = np.reshape(x_test,  x_test.shape  + (1,))
		y_test  = np.reshape(y_test,  y_test.shape  + (1,))

		print(x_train.shape)

		# Fit the model 
		self.model.fit(x_train, 
			           y_train, 
			           epochs=epochs, 
			           batch_size=batch_size,
                	   shuffle=True, 
                	   verbose=1,
                	   #callbacks=self.set_callbacks(),
                	   validation_data=(x_test, y_test))
	
	
	def set_callbacks(self):
		# Save the weights using a checkpoint.
		filepath = 'experiments/' + self.exp_name + '/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint]
		return callbacks_list

	def load_model(self): 
		############ TO DOOOOOOO ##############
		pass

	def getModel(self):
		return self.model

	def save_model(self):
		if not os.path.exists('experiments/' + self.exp_name + '/'):
			os.makedirs('experiments/' + self.exp_name + '/')


		model_json = self.model.to_json()
		with open('experiments/' + self.exp_name + '/model_' + self.model_name + '.json', "w") as json_file:
			json_file.write(model_json)

	def load_data(self, filename):
		return read_data(filename)

	def getData(self):
		return self.data