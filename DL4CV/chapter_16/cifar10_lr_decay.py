import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse


def step_decay(epoch):

    initAlpha = 0.01
    factor = 0.25
    dropEvery = 2
    alpha = initAlpha * (factor ** (np.floor(1 + epoch) / dropEvery))
    return float(alpha)


# ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", required=True)
# args = vars(ap.parse_args())

print("INFO loading cifar10 ...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# optimizer
callbacks = [LearningRateScheduler(step_decay)]
opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(32, 32, 3, 10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64,
              epochs=5, callbacks=callbacks, verbose=1)
model.save("minivggnet.h5")
# Evaluate

print("INFO evaluating model ...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=labelNames))


