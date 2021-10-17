from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout
from keras.layers import Flatten, Input, concatenate
from keras.models import Model
from keras import backend as K


class MiniGoogLeNet:
    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):

        """Defines a CONV > BN > RELU patterns"""
        """
        x > input layers to the function\n
        K = number of filters CONV layer is going to learn\n
        kX and kY: size of each of the K filters that will be learned\n
        
        """
        x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        return x

    @staticmethod
    def inception_module(x, numK1x1, numK3x3, chanDim):
        """Defines two CONV modules , then concatenate across the channel dimensions"""
        conv_1x1 = MiniGoogLeNet.conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
        conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)

        x = concatenate([conv_1x1, conv_3x3], axis=chanDim)

        return x

    @staticmethod
    def downsample_module(x, K, chanDim):
        """ defines the CONV module and POOL, then concatenates  across the channel dimensions"""
        conv_3x3 = MiniGoogLeNet.conv_module(x, K, 3, 3, (2, 2), chanDim, padding="valid")
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = concatenate([conv_3x3, pool], axis=chanDim)

        return x
    @staticmethod
    def qurish(width, height, depth, classes):

        input_shape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channel_first":
            input_shape = (depth, height, width)
            chanDim = 1


        # first module input
        inputs = Input(shape=input_shape)
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chanDim)

        # two inception modules followed by downsample module
        x = MiniGoogLeNet.inception_module(x, 32, 32, chanDim)
        x = MiniGoogLeNet.inception_module(x, 32, 48, chanDim)
        x = MiniGoogLeNet.downsample_module(x, 80, chanDim)

        # >>>>>>four inception module<<<<<<<<<<<<
        x = MiniGoogLeNet.inception_module(x, 112, 48, chanDim)
        x = MiniGoogLeNet.inception_module(x, 96, 64, chanDim)
        x = MiniGoogLeNet.inception_module(x, 80, 80, chanDim)
        x = MiniGoogLeNet.inception_module(x, 48, 96, chanDim)

        x = MiniGoogLeNet.downsample_module(x, 96, chanDim)

        # >>>>>>>>>>> two Inception modules followed by global POOL and Dropout
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        # >>>>>>>softmax classifier <<<<<<<<<<<
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        model = Model(inputs, x, name="googlenet")

        return model















