from pyimagesearch.nn.conv.dcgan import DCGAN
from keras.models import Model
from keras.layers import Input
from tensorflow.python.keras.optimizers import Adam
from keras.datasets import mnist
from sklearn.utils import shuffle
from imutils import build_montages
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-e", "--epochs", type=int, default=50)
ap.add_argument("-b", "--batch size for training")
args = vars(ap.parse_args())

NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]

print("INFO: loading MNIST dataset...")
((trainX, _), (testX, _)) = mnist.load_data()
trainImages = np.concatenate([trainX, testX])

trainImages = np.expand_dims(trainImages, axis=-1)
trainImages = (trainImages.astype("float") - 127.5) - 127.5

print("INFO: building generator...")
gen = DCGAN.build_generator(7, 64, channels=1)

print("INFO: building discriminator...")
disc = DCGAN.build_discriminator(28, 28, 1)
discOpt = Adam(lr=0.0002, beta_1=0.5, decay=0.0002 / NUM_EPOCHS)
disc.compile(loss="binary_crossentropy", optimizer=discOpt)

print("INfo: building GAN ... ")
disc.trainable = False
ganInput = Input(shape=(100,))
ganOutput = disc(gen(ganInput))
gan = Model(ganInput, ganOutput)

ganOpt = Adam(lr=0.0002, beta_1=0.5, decay=0.0002 / NUM_EPOCHS)
gan.compile(loss="binary_crossentropy", optimizer=discOpt)

print("INfo: starting training...")
benchmarkNoise = np.random.uniform(-1, 1, size=(256, 100))

for epoch in range(NUM_EPOCHS):
    print(f"INFO: starting epoch {epoch + 1} of {NUM_EPOCHS}...")
    batchesPerEpoch = int(trainImages.shape[0] / BATCH_SIZE)

    # loop over the batches
    for i in range(0, batchesPerEpoch):
        # empty output path
        p = None
        imageBatch = trainImages[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

        genImages = gen.predict(noise, verbose=0)

        # discriminator to recognize real and synthetic
        X = np.concatenate((imageBatch, genImages))
        y = ([1] * BATCH_SIZE) + ([0] * BATCH_SIZE)
        (X, y) = shuffle(X, y)

        discLoss = disc.train_on_batch(X, y)

        # train gan itself
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        ganLoss = gan.train_on_batch(noise, [1] * BATCH_SIZE)

        if i == batchesPerEpoch - 1:
            p = [args["output"], f"epoch_{str(epoch + 1).zfill(4)}_output.png"]

        else:
            if epoch < 10 and i % 25 == 0:
                p = [args["output"], f"epoch_{str(epoch + 1).zfill(4)}_step_{str(i).zfill(5)}.png"]

            elif epoch >= 10 and i % 100 == 0:
                p = [args["output"], f"epoch_{str(epoch + 1).zfill(4)}_step_{str(i).zfill(5)}.png"]

        if p is not None:
            print(f"INFO: Step {epoch+1}_{i}: discriminator_loss={discLoss}, "
                  f"adversarial_loss={ganLoss}")

        images = gen.predict(benchmarkNoise)
        images = ((images * 127.5) + 127.5).astype("uint8")
        images = np.repeat(images, 3, axis=-1)
        vis = build_montages(images, (28, 28), (16, 16))[0]

        p = os.path.sep.join(p)
        cv2.imwrite(p, vis)

















