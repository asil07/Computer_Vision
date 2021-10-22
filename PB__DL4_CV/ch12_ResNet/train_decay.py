import matplotlib
matplotlib.use("Agg")

from config import tiny_imagenet_config as cfg
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.nn.conv.resnet import ResNet
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import argparse
import json
import sys
import os

NUM_EPOCHS = 75
INIT_LR = 1e-1


def poly_decay(epoch):
    maxEpoch = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    alpha = baseLR * (1 - (epoch / float(maxEpoch))) ** power
    return alpha


ap = argparse.ArgumentParser()
ap.add_argument("-o", '--output', required=True)
ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=18, width_shift_range=0.2, height_shift_range=0.2,
                         zoom_range=0.15, shear_range=0.15, horizontal_flip=True,
                         fill_mode="nearest")

means = json.loads(open(cfg.DATASET_MEAN).read())

sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(cfg.TRAIN_HDF5, 64, preprocessors=[sp, mp, iap],
                                classes=cfg.NUM_CLASSES)
valGen = HDF5DatasetGenerator(cfg.VAL_HDF5, 64, preprocessors=[sp, mp, iap],
                              classes=cfg.NUM_CLASSES)

figPath = os.path.sep.join([args["output"], f"{os.getpid()}.png"])
jsonPath = os.path.sep.join([args["output"], f'{os.getpid()}.json'])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), LearningRateScheduler(poly_decay)]

print("INFO: compiling model...")
model = ResNet.build(64, 64, 3, cfg.NUM_CLASSES, stages=(3, 4, 6),
                     filters=(64, 128, 256, 512), reg=0.0005, dataset="tiny_imagenet")
opt = SGD(learning_rate=INIT_LR, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


print("Training ......")
model.fit(trainGen.generator(), steps_per_epoch=trainGen.numImages // 64,
          validation_data=valGen.generator(), validation_steps=valGen.numImages // 64,
          epochs=NUM_EPOCHS, max_queue_size=64 * 2,
          callbacks=callbacks, verbose=1)
print("saving...")
model.save(args['model'])

trainGen.close()
valGen.close()










