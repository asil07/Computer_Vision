from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from keras.datasets import cifar10
import argparse

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

print("INFO Compiling model ...")
opt = SGD(learning_rate=0.01,  momentum=0.09, nesterov=True)
model = MiniVGGNet.build(32, 32, 3, 10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# =============== Checkpoint ====================
checkpoint = ModelCheckpoint(args["weights"], monitor="val_loss", save_best_only=True, verbose=1)
callbacks = [checkpoint]

# ==================================================================================================
# train
print("INFO training network ...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=15, callbacks=callbacks, verbose=1)







