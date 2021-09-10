from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

import cv2
CATEGORIES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

def prepare(file_path):
    IMG_SIZE = 28
    img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = load_model("lenet_mnist.hdf5")

prediction = model.predict(prepare("number.jpg")).argmax(axis=1)
print(CATEGORIES[prediction[0]])


# def info(process):
#     print(f"[INFO] {process}")
#
#
# class_labels = ["cat", "dog", "panda"]
#
# orig = "number.jpg"
#
# img = image.load_img(orig, target_size=(28, 28))
# img = image.img_to_array(img)
# img = np.expand_dims(img, axis=0)
# img = img.astype("float") / 255.0
#
#
# info("Loading pre-trained model...")
# model = load_model("lenet_mnist.hdf5")
#
# info("predicting...")
# preds = model.predict(img, batch_size=128)[0]
# index = np.argmax(preds)
# # label = class_labels[index]
# print(f"Preds {preds}")
# print(f'argmax = {index}')
#
# image = cv2.imread(orig)
# for (n, p) in enumerate(preds):
#     cv2.putText(image, f"Label: , {preds.max()*100:.2f}%", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# cv2.imshow("Image", image)
# cv2.waitKey(0)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::
# def info(process):
#     print(f"[INFO] {process}")
#
# def input_prepare(img):
#     img = np.asarray(img)              # convert to array
#     img = cv2.resize(img, (28, 28))   # resize to target shape
#     img = cv2.bitwise_not(img)         # [optional] my input was white bg, I turned it to black - {bitwise_not} turns 1's into 0's and 0's into 1's
#     img = img / 255                    # normalize
#     img = img.reshape(-1, 784, 784, 1)          # reshaping
#     return img
#
# img = cv2.imread('number.jpg')
# orig = img.copy() # save for plotting later on
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # gray scaling
# img = input_prepare(img)
# print(img.shape)
#
# info("Loading pre-trained model...")
# model = load_model("lenet_mnist.hdf5")
#
# pred = model.predict(img)
# plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
# plt.title(np.argmax(pred, axis=1))
# plt.show()

# img = image.load_img(orig, target_size=(28, 28))
# img = image.img_to_array(img)
# img = np.expand_dims(img, axis=0)
# img = img.astype("float") / 255.0
#
#
#
#
# info("predicting...")
# preds = model.predict(img, batch_size=128).argmax(axis=1)
# index = np.argmax(preds)
#
# print(f"Preds {preds}")
# print(f'argmax = {index}')
#
# cv2.imshow("Image", image)
# cv2.waitKey(0)

