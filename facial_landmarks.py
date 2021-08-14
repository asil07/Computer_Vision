from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())
def result(object):
    cv2.imshow('asdas', object)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# initializes dlib's face detector (HOG-based) and then creates
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape-predictor')

image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(f'Image shape: {image.shape}')

while True:
    # detects image faces in the grayscale image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # converts dlib's rectangle to OpenCV bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # and draws rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # shows number
        cv2.putText(image, "Face #{}".format(i+1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Aniqlash", image)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break




