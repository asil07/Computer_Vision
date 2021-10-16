from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Activation
from keras.layers import Flatten, Input, add
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K


class ResNet:
    @staticmethod
    def residual_model(data, K, stride, chanDim, red=False,
                       reg=0.0001, bnEps=2e-5, bnMom=0.9):
        pass



