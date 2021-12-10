from conf import sr_config as config
from keras.models import load_model
from scipy import misc
import numpy as np
import argparse
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-b", "--baseline", required=True)
ap.add_argument("-o", "--output", required=True)
args = vars(ap.parse_args())

print("INFO loading model")
model = load_model(config.MODEL_PATH)

print("INFO: generateting image...")
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
w -= int(w % config.SCALE)
h -= int(h % config.SCALE)
image = image[0:h, 0:w]

scaled = misc.imresize(image, config.SCALE / 1.0, interp="bicubic")
cv2.imwrite(args["baseline"], scaled)

output = np.zeros(scaled.shape)
(h, w) = output.shape[:2]






