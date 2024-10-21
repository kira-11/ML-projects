import cv2
import mediapipe as mp
import numpy as np

mp_pose =  mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_holistics = mp.solutions.holistic
pose = mp_pose.Pose(model_complexity=0)

cap = cv2.VideoCapture("Video.mp4")#use your own file for testing pourposes


while True:
    ret, img = cap.read()
    img = cv2.resize(img,(600,400))


    results = pose.process(img)
    mp_draw.draw_landmarks(img,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
    cv2.imshow("pose estimation",img)

    h, w, c = img.shape
    opimg = np.zeros([h,w,c])
    opimg.fill(255)
    mp_draw.draw_landmarks(opimg,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
    cv2.imshow("extracted pose",opimg)



    print(results.pose_landmarks)
    if cv2.waitKey(1) & 0xFF==ord('d'):
        break
