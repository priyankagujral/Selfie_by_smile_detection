import numpy as np
import dlib
import cv2
import imutils
from imutils.video import VideoStream, FPS
from scipy.spatial import distance as dist
from collections import OrderedDict
import time

facial_landmark_indexes = OrderedDict([
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
    	("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 35)),
        ("jaw", (0, 17))
        ])

def shape_to_numpy(shape, dtype="int"):
    np_array = np.zeros((68,2), dtype = dtype)
    for i in range(0, 68):
        np_array[i] = (shape.part(i).x, shape.part(i).y)
    return np_array

def mouth_aspect_ratio(mouth):
    a_dist = dist.euclidean(mouth[2], mouth[10])
    b_dist = dist.euclidean(mouth[3], mouth[9])
    c_dist = dist.euclidean(mouth[4], mouth[8])
    d_dist = dist.euclidean(mouth[0], mouth[6])
    mar = (a_dist + b_dist + c_dist)/(3.0 * d_dist)
    return mar

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

mouth_close_thresh = 0.3
mouth_open_thresh = 0.38
cons_frames = 25
counter_frames = 0
total_smile = 0

vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)
fps= FPS().start()
cv2.namedWindow("test")
(start, end) = facial_landmark_indexes["mouth"]
    
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    for face in (faces):
        face_align = predictor(gray, face)
        face_align_numpy = shape_to_numpy(face_align)
        mouth_pts = face_align_numpy[start:end]
        mar = mouth_aspect_ratio(mouth_pts)
        
        if mar<=mouth_close_thresh or mar>mouth_open_thresh:
            counter_frames += 1
        else:
            if counter_frames >= cons_frames:
                total_smile += 1
                frame = vs.read()
                frame2 = frame.copy()
                time.sleep(0.6)
                print(counter_frames, mar)
                path = 'selfie/selfie_{}'+str(total_smile)+'.jpg'
                cv2.imwrite(path, frame)
            counter_frames = 0
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Selfies: "+str(total_smile), (10, 30), font, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "mar: "+str(mar), (300, 30), font, 0.5, (0, 255, 0), 2)
        mouthHull = cv2.convexHull(mouth_pts)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
    cv2.imshow("frame", frame)
    fps.update()
    
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
    
fps.stop()
cv2.destroyAllWindows()
vs.stop()
    
    







