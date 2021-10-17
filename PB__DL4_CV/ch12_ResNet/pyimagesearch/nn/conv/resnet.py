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
        shortcut = data

        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        # second CONV layers
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same",
                       use_bias=False, kernel_regularizer=l2(reg))(act2)

        # third conv block
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act3)

        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        x = add([conv3, shortcut])

        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e5, bnMom=0.9, dataset="cifar"):

        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)
            chanDim = 1

        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(inputs)

        if dataset == "cifar":
            x = Conv2D(filters[0], (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)


        for i in range(0, len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_model(x, K=filters[i+1], stride=stride, chanDim=chanDim,
                                      red=True, bnEps=bnEps, bnMom=bnMom)
            # loops over number of layers in the stage
            for j in range(0, stages[i] - 1):
                x = ResNet.residual_model(x, K=filters[i + 1], stride=(1, 1),
                                          chanDim=chanDim, bnEps=bnEps, bnMom=bnMom)

                x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
                x = Activation("relu")(x)
                x = AveragePooling2D((8, 8))(x)

                x = Flatten()(x)
                x = Dense(classes, kernel_regularizer=l2(reg))(x)
                x = Activation("softmax")(x)

                model = Model(inputs, x, name="resnet")

                return model





# stages = (2, 3, 4)
# for s in range(0, len(stages)):
#

