from pyimagesearch.nn.conv.dcgan import DCGAN
from keras.models import Model
from keras.layers import Input
from tensorflow.keras.optimizers import Adam
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









