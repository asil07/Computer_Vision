from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Input, concatenate
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K


class DeeperGoogLeNet:
    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim,
                    padding="same", reg=0.0005, name=None):

        (convName, bnName, actName) = (None, None, None)

        if name is not None:
            convName = name + "_conv"
            bnName = name + "_bn"
            actName = name + "_act"

        x = Conv2D(K, (kX, kY), strides=stride, padding=padding,
                   kernel_regularizer=l2(reg), name=convName)(x)
        x = BatchNormalization(axis=chanDim, name=bnName)(x)
        x = Activation("relu", name=actName)(x)

        return x

    @staticmethod
    def inception_module(x, num1x1, num3x3Reduce, num3x3, num5x5Reduce,
                         num5x5, num1x1Proj, chanDim, stage, reg=0.0005):

        # first branch
        first = DeeperGoogLeNet.conv_module(x, num1x1, 1, 1, (1, 1), chanDim, reg=reg,
                                            name=stage + "_first")
        # Second branch
        second = DeeperGoogLeNet.conv_module(x, num3x3Reduce, 1, 1, (1, 1), chanDim,
                                             reg=reg, name=stage + "_second1")
        second = DeeperGoogLeNet.conv_module(second, num3x3, kX=3, kY=3, stride=(1, 1),
                                             chanDim=chanDim, reg=reg, name=stage + "_second2")

        # third branch
        third = DeeperGoogLeNet.conv_module(x, num5x5Reduce, kX=1, kY=1, stride=(1, 1),
                                            chanDim=chanDim, reg=reg, name=stage + "_third1")
        third = DeeperGoogLeNet.conv_module(third, num5x5, kX=1, kY=1, stride=(5, 5),
                                            chanDim=chanDim, reg=reg, name=stage + "_third2")
        # Fourth branch
        fourth = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same", name=stage + "_pool")(x)

        fourth = DeeperGoogLeNet.conv_module(fourth, num1x1Proj, 1, 1, stride=(1, 1), chanDim=chanDim,
                                             reg=reg, name=stage + "_fourth")


        x = concatenate([first, second, third, fourth], axis=chanDim, name=stage + "_mixed")

        return x

    @staticmethod
    def build(width, height, depth, classes, reg=0.0005):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)
            chanDim = 1


        inputs = Input(shape=inputShape)

        x = DeeperGoogLeNet.conv_module(inputs, K=64, kX=5, kY=5, stride=(5, 5), chanDim=chanDim,
                                        reg=reg, name="block-1")
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool-1")(x)
        x = DeeperGoogLeNet.conv_module(x, 64, 1, 1, (1, 1), chanDim, reg=reg, name="block-2")
        x = DeeperGoogLeNet.conv_module(x, 192, 3, 3, (1, 1), chanDim, reg=reg, name="block-3")
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name="pool-2")(x)

        # applying two Inception modules
        x = DeeperGoogLeNet.inception_module(x=x, num1x1=64, num3x3Reduce=96, num3x3=128,
                                             num5x5Reduce=16, num5x5=32, num1x1Proj=32,
                                             chanDim=chanDim, stage="3a", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, num1x1=128, num3x3Reduce=128, num3x3=192,
                                             num5x5Reduce=32, num5x5=96, num1x1Proj=64,
                                             chanDim=chanDim, stage="3b", reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool-3")(x)










