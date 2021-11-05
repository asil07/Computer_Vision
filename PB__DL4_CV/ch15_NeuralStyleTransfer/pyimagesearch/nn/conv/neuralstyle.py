from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import cv2
import os


class NeuralStyle:
    def __init__(self, settings):

        self.S = settings

        (w, h) = load_img(self.S["input_path"]).size
        self.dims = (h, w)

        self.content = self.preprocess(settings["input_path"])
        self.style = self.preprocess(settings["style_path"])
        self.content = K.variable(self.content)
        self.style = K.variable(self.style)

        # allocates memory of output, combine three into single tensor
        self.output = K.placeholder((1, self.dims[0], self.dims[1], 3))
        self.input = K.concatenate([self.content, self.style, self.output], axis=0)

        print("INFO: laoding network...")
        self.model = self.S["net"](weights="imagenet", include_top=False,
                                   input_tensor=self.input)

        layerMap = {l.name: l.output for l in self.model.layers}

        contentFeatures = layerMap[self.S["content_layer"]]
        styleFeatures = contentFeatures[0, :, :, :]
        outputFeatures = contentFeatures[2, :, :, :]
        print(f"INFO: Style features:>> {styleFeatures}")

        contentLoss = self.featureReconLoss(styleFeatures, outputFeatures)
        contentLoss *= self.S["content_weight"]

        # initialize style loss (237)
        styleLoss = K.variable(0.0)
        weight = 1.0 / len(self.S["style_layers"])

        for layer in self.S["style_layers"]:

            # grabs current layer and uses it to extract the features
            styleOutput = layerMap[layer]
            styleFeatures = styleOutput[1, :, :, :]
            outputFeatures = styleOutput[2, :, :, :]

            # compute style reconstruction loss as we go
            T = self.styleReconLoss(styleFeatures, outputFeatures)
            styleLoss = styleLoss + (weight * T)

        styleLoss *= self.S["style_weight"]
        tvLoss = self.S["tv_weight"] * self.tvLoss(self.output)
        totalLoss = contentLoss + styleLoss + tvLoss

        grads = K.gradients(totalLoss, self.output)
        outputs = [totalLoss]
        outputs += grads

        # (238)
        self.lossAndGrads = K.function([self.output], outputs)

    def preprocess(self, p):
        image = load_img(p, target_size=self.dims)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        return image

    def deprocess(self, image):

        image = image.reshape((self.dims[0], self.dims[1], 3))
        image[:, :, 0] += 103.939
        image[:, :, 1] += 116.779
        image[:, :, 2] += 123.680

        image = np.clip(image, 0, 255).astype("uint8")

        return image

    def gramMat(self, X):

        """The gram matrix is the dot  product between the input  vectors
        and their respective transpose"""

        features = K.permute_dimensions(X, (2, 0, 1))
        features = K.batch_flatten(features)
        features = K.dot(features, K.transpose(features))

        return features

    def featureReconLoss(self, styleFeatures, outputFeatures):
        """The feature reconstruction loss is the squared error between the style features
        and output features"""
        return K.sum(K.square(outputFeatures - styleFeatures))

    def styleReconLoss(self, styleFeatures, outputFeatures):
        # A is the gram matrix for the style image and
        # G is the gram matrix for the generated image

        A = self.gramMat(styleFeatures)
        G = self.gramMat(outputFeatures)

        scale = 1.0 / float((2 * 3 * self.dims[0] * self.dims[1]) ** 2)
        loss = scale * K.sum(K.square(G - A))

        return loss

    def tvLoss(self, X):

        (h, w) = self.dims
        A = K.square(X[:, :h-1, :w-1, :] - X[:, 1:, :w-1, :])
        B = K.square(X[:, :h-1, :w-1, :] - X[:, :h-1, 1:, :])
        loss = K.sum(K.pow(A + B, 1.25))

        return loss

    def transfer(self, maxEvals=20):

        X = np.random.uniform(0, 255, (1, self.dims[0], self.dims[1], 3)) - 128

        for i in range(0, self.S['iterations']):
            print(f"INFO: starting iteration {i + 1} of {self.S['iterations']}")

            (X, loss, _) = fmin_l_bfgs_b(self.loss, X.flatten(), fprime=self.grads, maxfun=maxEvals)

            print(f"INFO: end of iterations {i + 1}, loss: {loss:.4e}")

            image = self.deprocess(X.copy())
            p = os.path.sep.join([self.S["output_path"], f"iter_{i}.png"])
            cv2.imwrite(p, image)

    def loss(self, X):
        X = X.reshape((1, self.dims[0], self.dims[1], 3))
        lossValue = self.lossAndGrads([X])[0]

        return lossValue

    def grads(self, X):
        X = X.reshape((1, self.dims[0], self.dims[1], 3))
        output = self.lossAndGrads([X])

        return output[1].flatten().astype(float)






















