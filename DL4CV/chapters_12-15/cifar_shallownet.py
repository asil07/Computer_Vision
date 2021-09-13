from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from pyimagesearch.nn.conv.shallownet import ShallowNet
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

print("INFO: Loading CIFAR10...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# Vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

print("INFO: Compiling model...")
opt = SGD(0.01)
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("INFO: training model...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=5, verbose=1)

print("INFO: Evaluating model...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="Train_Loss")
plt.plot(np.np.arange(0, 40), H.history["val_loss"], label="Val_loss")
plt.plot(np.np.arange(0, 40), H.history["accuracy"], label="Accuracy")
plt.plot(np.np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs #")
plt.ylabel("Loss / Acc")
plt.legend()
plt.show()










