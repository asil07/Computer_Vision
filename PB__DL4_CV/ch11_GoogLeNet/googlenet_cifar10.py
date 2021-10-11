import matplotlib

matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minigooglenet import MiniGoogLeNet
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os

NUM_EPOCHS = 2
INIT_LR = 5e-3
print(INIT_LR)


def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return alpha


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())


print("INFO: laoding CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
                         fill_mode="nearest")

figPath = os.path.sep.join([args["output"], f'{os.getpid()}.png'])
jsonPath = os.path.sep.join([args["output"], f"{os.getpid()}.json"])

callbacks = [TrainingMonitor(figPath, jsonPath, LearningRateScheduler(poly_decay))]


# compiling
print("INFO compiling model...")
opt = SGD(learning_rate=INIT_LR, momentum=0.9)
model = MiniGoogLeNet.qurish(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# training
print("INFO training network...")
model.fit(aug.flow(trainX, trainY, batch_size=64), validation_data=(testX, testY),
          steps_per_epoch=len(trainX) // 64, epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)

model.save(args['model'])












