from config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os
# import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True)
# args = vars(ap.parse_args())

trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[-1].split(".")[0] for p in trainPaths]

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# samples for test data from the training data
split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES,
                         stratify=trainLabels, random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

# samples for val data
split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_VAL_IMAGES,
                         stratify=trainLabels, random_state=42)

(trainPaths, valPaths, trainLabels, valLabels) = split

datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5)]
aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

# ++++++++++++++++++++++++++++++++
for (dType, paths, labels, outputPath) in datasets:
    print(f"INFO: building {outputPath}...")
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)

    # progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(),
               " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    # loop over image paths
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        image = aap.preprocess(image)

        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        writer.add([image], [label])
        pbar.update(i)

    pbar.finish()
    writer.close()

print("INFO: serializing means ... ")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()





