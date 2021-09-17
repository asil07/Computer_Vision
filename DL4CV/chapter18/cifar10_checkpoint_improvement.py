from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True)
args = vars(ap.parse_args())

print("INFO loading dataset ...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("INFO compiling model ...")
opt = SGD(learning_rate=0.01, momentum=0.09, nesterov=True)
model = MiniVGGNet.build(32, 32, 3, 10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# ==========CHECKPOINT===============
fname = os.path.sep.join([args["weights"], "weights - {epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
                             save_best_only=True, verbose=1)
callbacks = [checkpoint]
# train
print("INFO training model ...")
model.fit(trainX, trainY, validation_data=(testX, testY),
          batch_size=64, epochs=10, callbacks=callbacks, verbose=1)


