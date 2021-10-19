import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.resnet import ResNet
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import sys
import os


sys.setrecursionlimit(5000)

NUM_EPOCHS = 100
INIT_LR = 1e-1


def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return alpha


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-c", "--checkpoints", required=True)
ap.add_argument("-s", "--start-epoch", type=int, default=0)
args = vars(ap.parse_args())


print("[INFO]: loading cifar10...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype(float)
testX = testX.astype(float)

# apply mean
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# label binary
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                         horizontal_flip=True, fill_mode="nearest")

figPath = os.path.sep.join([args["output"], f"{os.getpid()}.png"])
jsonPath = os.path.sep.join([args["output"], f"{os.getpid()}.json"])

callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), LearningRateScheduler(poly_decay),
             EpochCheckpoint(args["checkpoints"], every=1, startAt=args["start_epoch"])]

print("info: compiling model...")
opt = SGD(learning_rate=INIT_LR, momentum=0.9)
model = ResNet.build(32, 32, 3, 10, stages=(9, 9, 9), filters=(64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("info: training network...")
model.fit(aug.flow(trainX, trainY, batch_size=128), validation_data=(testX, testY),
          steps_per_epoch=len(trainX) // 128, epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)

print("Info: seraializing model")
model.save(args["model"])



































