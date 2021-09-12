from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv.shallownet import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())


def info(process):
    print(f"[INFO] {process}")


info("loading images...")
imagePaths = list(paths.list_images(args['dataset']))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25,
                                                  random_state=42)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

info("compiling model...")
opt = SGD(0.005)
model = ShallowNet.build(32, 32, 3, 3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

info("training network ...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=32, epochs=100, verbose=1)
info("serializing network...")
model.save(args['model'])

info("evaluating network")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=["cat", "dog", "panda"]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="Training loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="Val loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="Training acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val acc")
plt.title("Training Loss and Acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss / Acc")
plt.legend()
plt.show()







