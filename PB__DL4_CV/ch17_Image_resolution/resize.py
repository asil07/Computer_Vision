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


for y in range(0, h - config.INPUT_DIM + 1, config.LABEL_SIZE):
    for x in range(0, w - config.INPUT_DIM + 1, config.LABEL_SIZE):
        crop = scaled[y:y + config.INPUT_DIM, x:x + config.INPUT_DIM]

        P = model.predict(np.expand_dims(crop, axis=0))
        P = P.reshape((config.LABEL_SIZE, config.LABEL_SIZE, 3))
        output[y + config.PAD:y + config.PAD + config.LABEL_SIZE,
        x + config.PAD:x + config.PAD + config.LABEL_SIZE] = P

output = output[config.PAD:h - ((h % config.INPUT_DIM) + config.PAD), config.PAD:w - ((w % config.INPUT_DIM) + config.PAD)]
output = np.clip(output, 0, 255).astype("uint8")

cv2.imwrite(args["output"], output)



