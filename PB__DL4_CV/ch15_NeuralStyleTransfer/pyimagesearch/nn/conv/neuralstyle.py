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
        self.style = self.preproceess(settings["style_path"])
        self.content = K.variable(self.content)
        self.style = K.variable(self.style)

        # allocates memory of output, combine three into single tensor
        self.output = K.placeholder((1, self.dims[0], self.dims[1], 3))
        self.input = K.concatenate([self.content, self.style, self.output], axis=0)

        print("INFO: laoding network...")
        self.model = self.S["net"](weights="imagenet", include_top=False,
                                   input_tensor=self.input)

        layerMap = {l.name: l.output for l in self.model.layers}
        print(f"INFO: {layerMap}")
        contentFeatures = layerMap[self.S["content_layer"]]
        styleFeatures = contentFeatures[0, :, :, :]
        outputFeatures = contentFeatures[2, :, :, :]
        print(f"INFO: Style features:>> {styleFeatures}")

        contentLoss = self.featreReconLoss(styleFeatures, outputFeatures)
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
            styleLoss += (weight * T)


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














