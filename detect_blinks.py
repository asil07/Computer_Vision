from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
    # Vertical landmarks of eye
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Horizontal landmarks of eye
    C = dist.euclidean(eye[0], eye[3])

    # Computes EAR
    ear = (A + B) / (2.0 * C)

    return ear


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False)
ap.add_argument("-v", "--video", type=str, default="")
args = vars(ap.parse_args())

# defines two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2

COUNTER = 0
TOTAL = 0

print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grabs the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")

# vs = FileVideoStream(args['video']).start()
# filestream = True
vs = VideoStream(0).start()
filestream = False
time.sleep(1.0)

while True:
    # if this is a file video stream, then needs to check if
    # there any more frames left in the buffer to process
    if filestream and not vs.more():
        break
    frame = vs.read()
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # chap va ong ko'zni koordinatalarini olamiz
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # va ularni EAR formulasiga solamiz
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        # ko'zlar contorlarini cv2.convexHull()  yordamida olamiz
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # va chizamiz
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        print(f"Kordinatalar {leftEyeHull[0][0]}")
        # let's check eye blink
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= 20:
                cv2.putText(frame, f"Uxlab qolyapsiz", (rightEyeHull[0][0][0] - 100, leftEyeHull[0][0][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, )
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # reset eye frame counter
            COUNTER = 0
        cv2.putText(frame, f"Ko'z yumilish soni: {TOTAL} ta", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, f"EAR: {ear}", (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Sanoq: {COUNTER}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()


