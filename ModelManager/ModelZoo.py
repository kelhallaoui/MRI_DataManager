from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K

def buildModel(model_name, input_shape):
	if model_name is 'basic':
		return model_basic(input_shape)
	elif model_name is 'autoencoder':
		return model_autoencoder(input_shape)

def model_basic(input_shape):
	pass

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
	return autoencoder

def model_encoder(input_shape):
	pass

def model_decoder(input_shape):
	pass


