from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K

def buildModel(model_name, input_shape):
	if   model_name is 'basic':            return model_basic(input_shape)
	elif model_name is 'inverted_ConvNet': return model_inverted_ConvNet(input_shape)
	elif model_name is 'autoencoder':      return model_autoencoder(input_shape)

def model_basic(input_shape):
	pass

def model_inverted_ConvNet(input_shape):
	""" Inverted ConvNet

	This model is the one used by Rosen for approximating the FFT.

	Args: 
		input_shape (tuple): The size of the inputs
	
	Returns:
		The inverted convnet model 
	"""
	input_shape = input_shape + (1,)
	img_size = 128 # Change this to be calculated from the input_shape

	input_img = Input(shape=input_shape)
	x = Dense(img_size**2, activation='tanh')(input_img)
	x = Dense(img_size**2, activation='tanh')(x)
	x = Reshape((img_size, img_size, 1))(x)
	x = Conv2D(64, (5, 5), activation='relu')(x)
	x = Conv2D(64, (5, 5), activation='relu')(x)
	x = Conv2DTranspose(64, (7,7), activation='relu', padding='valid', activity_regularizer = regularizers.l1(0.0001))(x)
	inv_ConvNet = ZeroPadding2D(padding=(1, 1), data_format=None)

	# Set a custom SGD optimizer
	sgd = optimizers.SGD(lr = 0.00002, decay = 0.9, momentum = 0, nesterov = False)
	model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
	return inv_ConvNet

def model_autoencoder(input_shape):
	input_shape = input_shape + (1,)

	input_img = Input(shape=input_shape)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	encoder = MaxPooling2D((2, 2), padding='same')(x)

	x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

	autoencoder = Model(input_img, decoder)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	return autoencoder

def model_encoder(input_shape):
	pass

def model_decoder(input_shape):
	pass


