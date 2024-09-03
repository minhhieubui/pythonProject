import cv2
import mediapipe as mp
import numpy as np
from djitellopy import tello
from time import sleep
from math import *

w, h = 360,240
pError_fb = 0
pError_ud = 0
pError_yaw = 0


# Kết nối với Drone
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
me.takeoff()
me.send_rc_control(0,0,20,0)
sleep(2.2)

# Khởi tạo camera
#cap = cv2.VideoCapture(0)

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    # Vẽ các đường nối
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    # Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        #print(id, lm)
        cx, cy , cz = int(lm.x * w), int(lm.y * h), int(-lm.z*500)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        if id == 0:
            info = [cx,cy,cz]
    return img,info

def trackFace( info,w, h, pError_fb,pError_ud ,pError_yaw):
    x, y, z = info[0],info[1], info[2]

    error_yaw = x - w // 2
    yaw = 0.5 * error_yaw + 0.5 * (error_yaw - pError_yaw)
    yaw = int(np.clip(yaw, -100, 100))

    error_fb = 350 - z
    fb = 0.25 * (error_fb) + 0.2 * (error_fb - pError_fb)
    fb = int(np.clip(fb, -100, 100))

    error_ud = -(y - h//2)
    ud = 0.6 * error_ud + 0.5 * (error_ud - pError_ud)
    ud = int(np.clip(ud, -100, 100))


    print(fb,ud,yaw)
    me.send_rc_control(0,fb,ud,yaw)
    return error_fb, error_ud, error_yaw

while True:
    #ret, frame = cap.read()

    frame = me.get_frame_read().frame

    # Chuyển đổi ảnh sang không gian màu RGB
    frame = cv2.resize(frame, (w, h))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện các điểm keypoint
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        #ghi nhận  thông số khung xương
        lm = make_landmark_timestep(results)
        lm_list.append(lm)

        # Vẽ Khung xương lên ảnh
        frame_rgb, info = draw_landmark_on_image(mpDraw, results, frame_rgb)

        # Tracking face
        pError_fb, pError_ud, pError_yaw = trackFace(info, w, h, pError_fb, pError_ud, pError_yaw)
    else :
        fb = 0
        ud = 0
        yaw = 0
        pError_fb = 0
        pError_ud = 0
        pError_yaw = 0
        print(fb, ud, yaw)
        me.send_rc_control(0,fb,ud,yaw)


    # Hiển thị frame
    cv2.imshow('image', frame_rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break

# Giải phóng bộ nhớ và dừng camera
#cap.release()
cv2.destroyAllWindows()