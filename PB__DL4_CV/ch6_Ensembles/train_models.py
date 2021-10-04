import matplotlib

matplotlib.use("Agg")
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to output directory")
ap.add_argument("-m", "--models", required=True,
                help="path to output models directory")
ap.add_argument("-n", "--num-models", type=int, default=5,
                help="# of models to train")
args = vars(ap.parse_args())

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                         horizontal_flip=True, fill_mode="nearest")

for i in np.arange(0, args["num_models"]):
    print(f"INFO: training model {i + 1}/{args['num_models']}")
    opt = SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(32, 32, 3, 10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                            validation_data=(testX, testY), epochs=5,
                            steps_per_epoch=len(trainX) // 64, verbose=1)

    p = [args["models"], f"model_{i}.model"]
    model.save(os.path.sep.join(p))

    # Evaluating
    prediction = model.predict(testX, batch_size=64)
    report = classification_report(testY.argmax(axis=1), prediction.argmax(axis=1),
                                   target_names=labelNames)
    p = [args["output"], f"model_{i}.txt"]
    f = open(os.path.sep.join(p), "w")
    f.write(report)
    f.close()

    p = [args["output"], f"model_{i}.png"]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 5), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 5), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 5), H.history["accuracy"], label="Train_Acc")
    plt.plot(np.arange(0, 5), H.history["val_accuracy"], label="Val_acc")
    plt.title("Training Loss and Accuracy for model {}".format(i))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(os.path.sep.join(p))
    plt.close()




