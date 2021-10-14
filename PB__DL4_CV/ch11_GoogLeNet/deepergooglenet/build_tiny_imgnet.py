from config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import os
import cv2

trainPaths = list(paths.list_images(config.TRAIN_IMAGES))
trainLabels = [p.split(os.path.sep)[-3] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# construct a testing set
split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES,
                         stratify=trainLabels, random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

# load validation and use these mapping
M = open(config.VAl_MAPPINGS).read().strip().split("\n")
M = [r.split("\t")[:2] for r in M]
valPaths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in M]
valLabels = le.transform([m[1] for m in M])

dataset = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5)
]

(R, G, B) = ([], [], [])

for (dtype, paths, labels, outputPath) in dataset:

    print(f"INFO: building {outputPath}")
    writer = HDF5DatasetWriter((len(paths), 64, 64, 3), outputPath)

    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(),
               " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    for (i, (path, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)

        # agar train dataset qurilyatgan bo'lsa rasmning har bir channeli "mean" qiymatini olamiz
        if dtype == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        writer.add([image], [label])
        pbar.update(i)

    pbar.finish()
    writer.close()

print("INFO: serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
with open(config.DATASET_MEAN, "w") as file:
    file.write(json.dumps(D))






