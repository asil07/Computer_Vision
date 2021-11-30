from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
from conf import sr_config as config
from imutils import paths
from scipy import misc
import shutil
import random
import cv2
import os

# creates output directries if not exist
for p in [config.IMAGES, config.LABELS]:
    if not os.path.exists(p):
        os.makedirs(p)

print("INFO: creating temporary images ... ")
imagePaths = list(paths.list_images(config.INPUT_IMAGES))
random.shuffle(imagePaths)
total = 0

for imagePath in imagePaths:
    image = cv2.imread(imagePath)

    (h, w) = image.shape[:2]
    w -= int(w % config.SCALE)
    h -= int(h % config.SCALE)
    image = image[0:h, 0:w]

    scaled = cv2.resize(image, 1.0 / config.SCALE, interpolation="bicubic")
    scaled = cv2.resize(scaled, config.SCALE/1.0, interpolation="bicubic")







