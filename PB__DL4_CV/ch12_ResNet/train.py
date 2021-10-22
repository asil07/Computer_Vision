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
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import argparse
import json
import sys

sys.setrecursionlimit(5000)

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True)
ap.add_argument("-m", "--model", type=str)
ap.add_argument("-s", "--start-epoch", type=int, default=0)

args = vars(ap.parse_args())


aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15, width_shift_range=0.2,
                         height_shift_range=0.2, shear_range=0.15, horizontal_flip=True,
                         fill_mode="nearest")

means = json.loads(open(cfg.DATASET_MEAN).read())

sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(cfg.TRAIN_HDF5, 64, aug=aug, preprocessors=[sp, mp, iap], classes=cfg.NUM_CLASSES)
valGen = HDF5DatasetGenerator(cfg.VAL_HDF5, 64, preprocessors=[sp, mp, iap], classes=cfg.NUM_CLASSES)

if args["model"] is None:
    print("INFO: Compiling...")
    model = ResNet.build(64, 64, 3, cfg.NUM_CLASSES, stages=(3, 4, 6), filters=(64, 128, 256, 512), reg=0.0005,
                         dataset="tiny_imagenet")
    opt = SGD(learning_rate=1e-1, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

else:
    print(f"INFO: Loading {args['model']}")
    model = load_model(args['model'])

    print(f"INFO: old learning rate: {K.get_value(model.optimizer.lr)}")
    K.set_value(model.optimizer.lr, 1e-5)
    print(f"INFO: New learning rate: {K.get_value(model.optimizer.lr)}")

callback = [
    EpochCheckpoint(args["checkpoints"], every=1, startAt=args["start_epoch"]),
    TrainingMonitor(cfg.FIG_PATH, jsonPath=cfg.JSON_PATH, startAt=args["start_epoch"])
]

model.fit(trainGen.generator(), steps_per_epoch=trainGen.numImages // 64,
          validation_data=valGen.generator(), validation_steps=valGen.numImages // 64,
          epochs=50, max_queue_size=64 * 8, callbacks=callback, verbose=1)

trainGen.close()
valGen.close()






