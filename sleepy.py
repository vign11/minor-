import cv2
from scipy.spatial import distance
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.FaceDetectionModule import FaceDetector
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
from collections import OrderedDict

number = 0

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

def cal_MAR(mouth):
    dist_x = distance.euclidean(mouth[0], mouth[6])
    dist_y = distance.euclidean(mouth[3], mouth[9])
    mar = dist_y / dist_x
    return mar

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])

cap = cv2.VideoCapture(0)
width = cap.get(3)
height = cap.get(4)
print(f"Current Camera Resolution: {width}x{height}")

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

detector = FaceMeshDetector()
detector2 = FaceDetector()
blink_count = 0
yawn_count = 0
blink_thresh = 3
yawn_thresh = 3

# Drowsiness detection parameters
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 40
EYE_AR_CONSEC_FRAMES_blink = 18
COUNTER = 0

# Yawn detection parameters
YAWN_AR_THRESH = 0.5
YAWN_AR_CONSEC_FRAMES = 15
COUNTER_YAWN = 0
total_blinks = 0
total_yawns = 0
EAR=0
while True:
    try:
        ret, frame = cap.read()
        #print("Webcam read successful?", ret)
        if not ret:
            print("Can't receive frame (stream end?). Exiting....")
            break
        frame, faces = detector.findFaceMesh(frame, draw=False)
        frame, bboxs = detector2.findFaces(frame, draw=False)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = None
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            l, r = (x, y), (x + w, y + h)
            w = distance.euclidean(l, r)
            W, f = 6.3, 825
            d = (W * f) / w
            try:
                cv2.putText(frame, "Distance: {}cm".format(int(d)), (400, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
            except Exception as error:
                print(error)

        faces = hog_face_detector(gray)
        for face in faces:
            face_landmarks = dlib_facelandmark(gray, face)
            face_landmarks = face_utils.shape_to_np(face_landmarks)
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                for (x, y) in face_landmarks[i:j]:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            roi = frame[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=600, inter=cv2.INTER_CUBIC)
            leftEye = []
            rightEye = []
            mouth = face_landmarks[mStart:mEnd]
            if len(face_landmarks) >= 48:
                for n in range(36,42):
                    x,y = face_landmarks[n]
                    leftEye.append((x,y))
                    next_point = n + 1
                    if n == 41:
                        next_point = 36
                    x2 ,y2 = face_landmarks[next_point]
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
                for n in range(42,48):
                    x,y = face_landmarks[n]
                    rightEye.append((x,y))
                    next_point = n + 1
                    if n == 47:
                        next_point = 42
                    x2 ,y2 = face_landmarks[next_point]
                cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)
            mar = cal_MAR(mouth)

            EAR = (left_ear + right_ear) / 2
            leftEye = np.array(leftEye, dtype=np.int32).reshape((-1, 1, 2))
            rightEye = np.array(rightEye, dtype=np.int32).reshape((-1, 1, 2))
            mouth = np.array(mouth, dtype=np.int32).reshape((-1, 1, 2))
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            
            if EAR < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES_blink:
                    cv2.putText(frame, "YOU JUST BLINKED", (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    total_blinks += 1
                COUNTER = 0
                cv2.putText(frame, "YOU ARE AWAKE", (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2)
                #rather than this blink count.....

            cv2.putText(frame, "EAR: {:.2f}".format(EAR), (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Blink count: {}".format(total_blinks), (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if mar > YAWN_AR_THRESH:
                COUNTER_YAWN += 1
                
                if COUNTER_YAWN >= YAWN_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "YOU ARE YAWNING", (400, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2)
                    total_yawns += 1
            else:
                COUNTER_YAWN = 0
        cv2.putText(frame, "Yawn count: {}".format(total_yawns), (400, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        EAR = round(EAR,2)
        #i think here it is more accurate
        if EAR < 0.26:
            cv2.putText(frame, "DROWSY", (195,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
            print("Drowsy")
            print(EAR)
        key = cv2.waitKey(1)
        if key == ord('c'):
            number += 1
            print(f"Image {number} has been captured")
            cv2.imwrite(f'image{number}.png', frame.copy())
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

        cv2.imshow("Image", frame)

    except Exception as e:
        print(f"An error occurred: {e}")
        break

cap.release()
cv2.destroyAllWindows()

 