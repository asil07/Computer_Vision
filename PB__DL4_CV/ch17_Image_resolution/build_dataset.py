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

    for y in range(0, h - config.INPUT_DIM +1, config.STRIDE):
        for x in range(0, w- config.INPUT_DIM + 1, config.STRIDE):

            crop = scaled[y:y + config.INPUT_DIM, x:x + config.INPUT_DIM]

            target = image[
                y + config.PAD:y + config.PAD + config.LABEL_SIZE,
                x + config.PAD:x + config.PAD + config.LABEL_SIZE
            ]
            crop_path = os.path.sep.join([config.IMAGES, f"{total}.png"])
            target_path = os.path.sep.join([config.LABELS, f"{total}.png"])

            cv2.imwrite(crop_path, crop)
            cv2.imwrite(target_path, target)

            total += 1

print("INFO: building HDF5 datasets...")
inputPaths = sorted(list(paths.list_images(config.IMAGES)))
outputPaths = sorted(list(paths.list_images(config.LABELS)))

inputWriter = HDF5DatasetWriter((len(inputPaths), config.INPUT_DIM, config.INPUT_DIM, 3), config.INPUT_DB)
outputWriter = HDF5DatasetWriter((len(outputPaths), config.LABEL_SIZE, config.LABEL_SIZE, 3), config.OUTPUT_DB)

for (inputPath, outputPath) in zip(inputPaths, outputPaths):
    inputImage = cv2.imread(inputPath)
    outputImage = cv2.imread(outputPath)
    inputWriter.add([inputImage], [-1])
    outputWriter.add([outputImage], [-1])

inputWriter.close()
outputWriter.close()

print("INFO: Cleaning up...")
shutil.rmtree(config.IMAGES)
shutil.rmtree(config.LABELS)
















