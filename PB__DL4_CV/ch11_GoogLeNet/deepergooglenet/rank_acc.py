from config import tiny_imagenet_config as cfg
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.utils.ranked import rank5_accuracy
from  pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from keras.models import load_model
import json


means = json.loads(open(cfg.DATASET_MEAN).read())

sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

testGen = HDF5DatasetGenerator(cfg.TEST_HDF5, 64, preprocessors=[sp, mp, iap],
                               classes=cfg.NUM_CLASSES)
print("INFO: loading model...")
model = load_model(cfg.MODEL_PATH)

print("INFO: predicting on test data...")
predictions = model.predict_generator(testGen.generator(), steps=testGen.numImages // 64,
                                      max_queu_size=64*2)
(rank1, rank5) = rank5_accuracy(predictions, testGen.db["labels"])
print(f"Rank-1: {rank1 * 100:.2f}")
print(f"Rank-5: {rank5 * 100:.2f}")

testGen.close()



