from keras.models import Sequential
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Flatten, Dense, Reshape


class DCGAN:
    @staticmethod
    def build_generator(dim, depth, channels=1, inputDim=100, outputDim=512):

        model = Sequential()
        inputShape = (dim, dim, depth)
        chanDim = -1

        # first set of FC => RELU > BN layers
        model.add(Dense(input_dim=inputDim, units=outputDim))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        # second set
        model.add(Dense(dim * dim * depth))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        # reshaping output of previous layer set
        model.add(Reshape(inputShape))
        model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        # another upsample
        model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation("tanh"))

        return model

    @staticmethod
    def build_discriminator(width, height, depth, alpha=0.2):
        model = Sequential()
        inputShape = (height, width, depth)

        model.add(Conv2D(32, (5, 5), padding="same", strides=(2, 2),
                         input_shape=inputShape))
        model.add(LeakyReLU(alpha=alpha))

        # second set
        model.add(Conv2D(64, (5, 5), padding="same", strides=(2, 2),
                         input_shape=inputShape))
        model.add(LeakyReLU(alpha=alpha))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha))

        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        return model

    


