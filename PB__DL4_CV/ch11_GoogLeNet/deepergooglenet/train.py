import matplotlib
matplotlib.use("Agg")

from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.nn.conv.deepergooglenet import DeeperGoogLeNet
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from keras.models import load_model
import keras.backend as K
import argparse
import json


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True)
ap.add_argument("-m", "--model", type=str)
ap.add_argument("-s", "--start-epoch", type=int, default=0)
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15, width_shift_range=0.2,
                         height_shift_range=0.2, shear_range=0.15, horizontal_flip=True,
                         fill_mode='nearest')
means = json.loads(open(config.DATASET_MEAN).read())

# preprocessors
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# training and validation dataset
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug=aug,
                                preprocessors=[sp, mp, iap],
                                classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64, preprocessors=[sp, mp, iap],
                              classes=config.NUM_CLASSES)

# compile if model not given
if args['model'] is None:
    print("INFO: compiling model...")
    model = DeeperGoogLeNet.build(width=64, height=64, depth=3, classes=config.NUM_CLASSES,
                                  reg=0.0002)
    opt = SGD(learning_rate=1e-2, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
    print("INFO: loading model ...")
    model = load_model(args['model'])
    print(f"INFO: old learning rate {K.get_value(model.optimizer.lr)}")
    K.set_value(model.optimizer.lr, 1e-5)
    print(f"INFO: new learning rate {K.get_value(model.optimizer.lr)}")

callbacks = [
    EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
    TrainingMonitor(config.FIG_PATH, jsonPath=config.JSON_PATH, startAt=args["start_epoch"])
]

model.fit(trainGen.generator(), steps_per_epoch=trainGen.numImages // 64,
          validation_data=valGen.generator(), validation_steps=valGen.numImages // 64,
          epochs=10, max_queue_size=64 * 2, callbacks=callbacks, verbose=1)

trainGen.close()
valGen.close()




