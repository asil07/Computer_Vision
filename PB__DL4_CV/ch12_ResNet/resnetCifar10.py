import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.resnet import ResNet
from pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
from keras.datasets import cifar10
from keras.models import load_model
import keras.backend as K
import numpy as np
import argparse
import sys

sys.setrecursionlimit(5000)

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True)
ap.add_argument("-m", "--model", type=str)
ap.add_argument("-s", "--start-epoch", type=int, default=0)
args = vars(ap.parse_args())

print("INFO: laoding dataset cifar10...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


aug = ImageDataGenerator(height_shift_range=0.1, width_shift_range=0.1, horizontal_flip=True,
                         fill_mode="nearest")

if args['model'] is None:
    print("INFO: compiling model....")
    opt = SGD(learning_rate=1e-1)
    opt_A = Adam(learning_rate=0.1)
    model = ResNet.build(width=32, height=32, depth=3, classes=10,
                         stages=(9, 9, 9), filters=(16, 64, 128, 256), reg=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=opt_A, metrics=["accuracy"])

else:
    print(f"INFO loading {args['model']}...")
    model = load_model(args["model"])

    print(f"INFO old-learning rate: {K.get_value(model.optimizer.lr)}")
    K.set_value(model.optimizer.lr, 1e-5)
    print(f"INFo new learning rate {K.get_value(model.optimizer.lr)}")

callbacks = [
    EpochCheckpoint(args["checkpoints"], every=1, startAt=args["start_epoch"]),
    TrainingMonitor("output/resnet56_cifar10.png", jsonPath="output/resnet56cifar.json",
                    startAt=args['start_epoch'])

]

# ===================================
print("Info: training network...")
model.fit(aug.flow(trainX, trainY, batch_size=128), validation_data=(testX, testY),
          steps_per_epoch=len(trainX) // 128, epochs=10, callbacks=callbacks, verbose=1)


