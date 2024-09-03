import cv2
from djitellopy import tello
import time
import KeyPressModule as kp
global img_default
import numpy as np
from math import *

kp.init()
#e = tello.Tello()
#me.connect()
#print(me.get_battery())
#me.streamon()
fspeed  = 12       # toc do 15m/s
aspeed = 36       # toc do 50 do/s
T = 0.25
dT = fspeed * T
aT = aspeed * T
x,y = 600,400
a = 0
yaw = 0
points = [(0,0),(0,0)]

def getkeyboardInput():
    lf, fb, ud, yv = 0, 0, 0, 0
    global yaw,x,y,a
    dspeed = 50
    aspeed = 50
    d = 0

    if kp.getKey("LEFT"):
        lf = -dspeed
        d = dT
        a = -180

    elif kp.getKey("RIGHT"):
        lf = dspeed
        d = dT
        a = 0

    if kp.getKey("UP"):
        fb = dspeed
        d = dT
        a = -90

    elif kp.getKey("DOWN"):
        fb = -dspeed
        d = dT
        a = 90

    if kp.getKey("w"):
        ud = dspeed
    elif kp.getKey("s"):
        ud = -dspeed

    if kp.getKey("a"):
        yv = aspeed
        yaw -= aT

    elif kp.getKey("d"):
        yv = -aspeed
        yaw += aT
    time.sleep(0.25)
    a += yaw
    x += int(d * cos(radians(a)))
    y += int(d * sin(radians(a)))

    x1 = int(x + 10*cos(radians(yaw-90)))
    y1 = int(y + 10*sin(radians(yaw-90)))

    #if kp.getKey("q"): me.land()
    #if kp.getKey("e"): me.takeoff()
    if kp.getKey("z"):
        cv2.imwrite(f'Resources/Images/{time.time()}.jpg',img_default)
        time.sleep(0.3)

    return[lf, fb, ud, yv, x, y,x1,y1]

def drawPoint(img,points,huong):
    for point in points:
        cv2.circle(img,point,5,(0,0,255), cv2.FILLED)
    cv2.circle(img, points[-1], 10, (255, 0, 255), cv2.FILLED)
    cv2.circle(img, huong, 5, (255, 0, 0), cv2.FILLED)
    cv2.putText(img,f'({(points[-1][0]-600)/100},{(points[-1][1]-400)/100})m',(points[-1][0]+10,points[-1][1]+50),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1 )

while True:
    vals = getkeyboardInput()
    #me.send_rc_control(vals[0],vals[1],vals[2],vals[3])
    img = np.zeros((800,1200,3), np.uint8)
    huong = (vals[6],vals[7])
    if points[-1][0] != vals[4] or points[-1][1] != vals[5]:
        points.append((vals[4],vals[5]))
    drawPoint(img,points,huong)

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #me.land()
        cv2.destroyAllWindows()
        break



