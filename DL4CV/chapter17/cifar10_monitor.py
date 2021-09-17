import matplotlib
matplotlib.use("Agg")

from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from tensorflow.keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True)
args = vars(ap.parse_args())

print(f"INFO process ID: {os.getpid()}")

print("INFO loading cifar10 dataset ...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

print("INFO compiling model, optimizer ...")
opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(32, 32, 3, 10)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=opt)


# ++++++++++++++++++++ Constructs the set of callbacks +++++++++++++++++++++++++
figPath = os.path.sep.join( [args['output'], "{}.png".format(os.getpid())] )
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])

callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

print("INFO model training starts ...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=5,
          callbacks=callbacks, verbose=1)





