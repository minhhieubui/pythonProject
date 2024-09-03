import cv2
from djitellopy import tello
import time
import KeyPressModule as kp
global img_default
import numpy as np
from math import *

# kp.init()

# me = tello.Tello()
# me.connect()
# print(me.get_battery())
# me.streamon()

fspeed  = 40       # toc do 15 cm/s
aspeed = 36       # toc do 50 do/s
T = 0.25
dT = fspeed * T
aT = aspeed * T
x,y = 150,750
a = 0
yaw = 0
lr = 0
fb = 0
i = 0
points = [(0,0),(150,750)]
start_point = (150,700)
#end_points = [(50,750),(50,700)]

#thong so khung
cd = 600  #chieu dai khung
cr = 100  #chieu rong khung
kc = 100  #khoảng cách giữa 2 khung
n = 5   # số khung, mỗi khung có 2 dãy
hang = 7
vt = 200  # vị trí muốn tới

def flytoPoint():
    global yaw,x,y,a,lr,fb,i
    d = 0
    line = hang // 2
    du = hang % 2
    d_lr = (2*line-1)*100

    if lr < d_lr:
        d = dT
        lr += d
        a = 0
    elif lr < -d_lr:
        d = dT
        lr += d
        a = -180
    elif fb < vt+50:
        fb += dT
        d = dT
        a = -90
    elif du == 1 and i < 10:
        i +=1
        yaw += 9
    elif du == 0 and i < 10:
        i +=1
        yaw += -9



    time.sleep(0.25)
    a += yaw
    x += int(d * cos(radians(a)))
    y += int(d * sin(radians(a)))

    # huong của drone
    x1 = int(x + 10*cos(radians(yaw-90)))
    y1 = int(y + 10*sin(radians(yaw-90)))
    return [x,y,x1,y1]






def drawMap(img,start_point):
    for i in range(n):
        cv2.line(img, start_point, (start_point[0],start_point[1]-cd), (0,0,255), thickness=5)
        cv2.putText(img, f'{i * 2 + 1}', (start_point[0] - 50, start_point[1] - cd - 50), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 0), 1)
        cv2.putText(img, f'{i * 2 + 2}', (start_point[0] + 50, start_point[1] - cd - 50), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 0), 1)
        start_point = (start_point[0]+cr+kc,start_point[1])


def drawPoint(img,points,huong):
    for point in points:
        cv2.circle(img,point,1,(0,255,0), cv2.FILLED)
    cv2.circle(img, points[-1], 10, (255, 0, 255), cv2.FILLED)
    cv2.circle(img, huong, 5, (255, 0, 0), cv2.FILLED)
    cv2.putText(img,f'({(points[-1][0]-150)/100},{(points[-1][1]-700)/100})m',(points[-1][0]+10,points[-1][1]+30),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1 )

while True:
    vals = flytoPoint()
#    me.send_rc_control(vals[0],vals[1],vals[2],vals[3])
    img = np.zeros((800,1200,3), np.uint8)
    drawMap(img,start_point)
    huong = (vals[2], vals[3])
    if points[-1][0] != vals[0] or points[-1][1] != vals[1]:
        points.append((vals[0],vals[1]))
    drawPoint(img,points,huong)


    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #me.land()
        cv2.destroyAllWindows()
        break



