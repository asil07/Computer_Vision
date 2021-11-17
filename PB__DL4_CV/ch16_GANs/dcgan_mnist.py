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













