import matplotlib
matplotlib.use("Agg")

from config import tiny_imagenet_config as cfg
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.nn.conv.resnet import ResNet
from tensorflow.keras.optimizers import SGD
from keras.models import load_model
import keras.backend as K
import argparse
import json
import sys

sys.setrecursionlimit(5000)


