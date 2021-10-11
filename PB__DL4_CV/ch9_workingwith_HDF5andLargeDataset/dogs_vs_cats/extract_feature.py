from tensorflow.keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-bs", "--batch-size", type=int, default=16)
ap.add_argument("-s", "--buffer-size", type=int, default=1000)
args = vars(ap.parse_args())

bs = args["batch_size"]
# load images
print("INFO: Loading images... ")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# loading Network
print("INFO: loading network... ")
model = ResNet50(weights="imagenet", include_top=False)

dataset = HDF5DatasetWriter((len(imagePaths), 2048), args["output"], dataKey="features",
                            bufSize=args["buffer_size"])
dataset.storeClassLabels(le.classes_)

# progressbar
widgets = ["Extracting Features: ", progressbar.Percentage(), "", progressbar.Bar(), "",
           progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# images in batches
for i in np.arange(0, len(imagePaths), bs):
    # extracts the batch of images and labels, then initializes the
    # list of actual images that will be passed through the network
    # for feature extraction
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i+bs]
    batchImages = []
    # images and labels in current batch
    for (j, imagePath) in enumerate(batchPaths):
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        batchImages.append(image)

    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)
    print(f'Features[0] >>> {features.shape[0]}')
    print(f'Features.shape {features.shape}')
    # reshapes the features so that each image is represented by
    # a flattened feature vector of the ‘MaxPooling2D‘ outputs
    features = features.reshape((features.shape[0], 2048))

    dataset.add(features, batchLabels)
    pbar.update(i)
dataset.close()
pbar.finish()






