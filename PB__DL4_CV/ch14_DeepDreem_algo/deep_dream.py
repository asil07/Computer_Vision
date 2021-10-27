from tensorflow.keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K
from scipy import ndimage
import numpy as np
import argparse
import cv2
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def preprocess(p):

    image = load_img(p)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    return image


def deprocess(image):

    img = image.reshape((image.shape[1], image.shape[2], 3))

    img /= 2.0
    img += 0.5
    img *= 255.0
    img = np.clip(img, 0, 255).astype("uint8")

    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return image


def resize_img(image, size):
    resized = np.copy(image)
    resized = ndimage.zoom(resized, (1, float(size[0]) / resized.shape[1],
                                     float(size[1]) / resized.shape[2], 1), order=1)

    return resized


def eval_loss_and_gradient(X):
    """Returns a tuple of the loss and gradient"""

    output = fetchLossGrads([X])
    (loss, G) = (output[0], output[1])

    return loss, G

def gradient_ascent(X, iters, alpha, maxLoss=-np.inf):

    for  i in range(0, iters):
        (loss, G) = eval_loss_and_gradient(X)

        if loss > maxLoss:
            break

        print(f"INFO: Loss at {i}: {loss}")
        X += alpha * G

    return X


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-o", "--output", required=True)
args = vars(ap.parse_args())

LAYERS = {"mixed2": 2.0,
          "mixed3": 0.5}

NUM_OCTAVES = 3
OCTAVE_SCALE = 1.4
ALPHA = 0.001
NUM_ITER = 5
MAX_LOSS = 10.00


# responsible for not to be updated weight if any layer during dreaming
K.set_learning_phase(0)

print("INFO: Loadingm inception model...")
model = InceptionV3(weights="imagenet", include_top=False)
dream = model.input

loss = K.variable(0.0)
layerMap = {layer.name: layer for layer in model.layers}

for layerName in LAYERS:

    x = layerMap[layerName].output
    coeff = LAYERS[layerName]
    scaling = K.prod(K.cast(K.shape(x), "float32"))
    loss = loss + ( coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling)


grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

outputs = [loss, grads]
fetchLossGrads = K.function([dream], outputs)

image = preprocess(args['image'])
dims = image.shape[1:3]

octaveDims = [dims]

for i in (1, NUM_OCTAVES):
    size = [int(d / (OCTAVE_SCALE ** i)) for d in dims]
    octaveDims.append(size)

octaveDims = octaveDims[::-1]

orig = np.copy(image)
shrunk = resize_img(image, octaveDims[0])

for (o, size) in enumerate(octaveDims):
    print(f"INFO: starting actave {o}...")
    image = resize_img(image, size)
    image = gradient_ascent(image, iters=NUM_ITER, alpha=ALPHA, maxLoss=MAX_LOSS)

    upscaled = resize_img(shrunk, size)
    downscaled = resize_img(orig, size)

    lost = downscaled - upscaled

    image += lost
    shrunk = resize_img(orig, size)

image = deprocess(image)
cv2.imwrite(args["output"], image)







