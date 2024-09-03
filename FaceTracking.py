import cv2
from djitellopy import tello
import numpy as np
from time import sleep
import math

'''
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
me.takeoff()
me.send_rc_control(0,0,20,0)
sleep(2.2)
'''

fbRange =[6200,7000]
pid = [0.5, 0.4, 0]
w, h = 360,240
pError_fb = 0
pError_ud = 0
pError_yaw = 0

cap = cv2.VideoCapture(0)

def findFace(img):
    faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2,8)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cx = x + w//2
        cy = y + h//2
        area = w*h
        cv2.circle(img,(cx,cy),5,(0,255,0),cv2.FILLED)
        myFaceListC.append([cx,cy])
        myFaceListArea.append(area)
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i],myFaceListArea[i]]
    else:
        return img, [[0,0],0]

def trackFace( info, w, pid, pError_fb,pError_ud ,pError_yaw):
    area = info[1]
    x, y = info[0]

    error_yaw = x - w // 2
    yaw = pid[0] * error_yaw + pid[1] * (error_yaw - pError_yaw)
    yaw = int(np.clip(yaw, -100, 100))

    if area > fbRange[0] and area < fbRange[1]:
        error_fb = 0
    elif area > fbRange[1]:
        error_fb = - math.sqrt(area - fbRange[1])
    elif area < fbRange[0] and area != 0:
        error_fb = math.sqrt(fbRange[0] - area)
    else: error_fb = 0
    fb = 0.4 * (error_fb) + 0.4 * (error_fb - pError_fb)
    fb = int(np.clip(fb, -100, 100))

    error_ud = -(y - h//2)
    ud = 0.6 * error_ud + 0.5 * (error_ud - pError_ud)
    ud = int(np.clip(ud, -100, 100))

    if x == 0:
        fb = 0
        ud = 0
        yaw = 0
        error_fb = 0
        error_ud = 0
        error_yaw = 0

    print(fb,ud,yaw)
    #me.send_rc_control(0,fb,ud,yaw)
    return error_fb, error_ud, error_yaw

while True:
    _, img = cap.read()
    #img_default = me.get_frame_read().frame
    #img = cv2.cvtColor(img_default, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(w,h))
    img, info = findFace(img)
    pError_fb,pError_ud, pError_yaw = trackFace(info, w, pid, pError_fb, pError_ud, pError_yaw)
    cv2.imshow("output",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #me.land()
        cv2.destroyAllWindows()
        break

