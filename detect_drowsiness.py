from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2


def play_sound(path):
    playsound.playsound(path)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[3])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True)
ap.add_argument("-a", "--alarm", type=str, default="")
# ap.add_argument("-w", "--webcam", type=str, default=0)

args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSECS_FRAMES = 20
COUNTER = 0
ALARM_ON = False

print("[Ma'lumot] Yuzni aniqlovchilar yuklanmoqda...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, LEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

print("[Ma'lumot] Jonli Video yoqilmoqda...")
vs = VideoStream(0).start()
time.sleep(1)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:LEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # O'rtacha qiymati ikkala  ko'z EAR ning
        ear = (leftEAR + rightEAR) / 2.0

        # Visualize
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSECS_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    # checks to see if an alarm file was supplied,
                    # and if so, starts a thread to have the alarm
                    # sound played in the background
                    if args["alarm"] != "":
                        t = Thread(target=play_sound, args=(args["alarm"],))
                        t.daemon = True
                        t.start()
                cv2.putText(frame, "OGOHLANTIRUVCHI SIGNAL", (rightEyeHull[0][0][0] - 100, leftEyeHull[0][0][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 4)
        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Counter: {COUNTER}", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()








