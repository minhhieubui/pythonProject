import cv2
from djitellopy import tello
import time
import KeyPressModule as kp
global img_default

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
#me.takeoff()

def getkeyboardInput():
    lf, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if kp.getKey("LEFT"): lf = speed
    elif kp.getKey("RIGHT"): lf = -speed

    if kp.getKey("UP"): fb = speed
    elif kp.getKey("DOWN"): fb = -speed

    if kp.getKey("w"): ud = speed
    elif kp.getKey("s"): ud = -speed

    if kp.getKey("a"): yv = speed
    elif kp.getKey("d"): yv = -speed

    if kp.getKey("q"): me.land()
    if kp.getKey("e"): me.takeoff()
    if kp.getKey("z"):
        cv2.imwrite(f'Resources/Images/{time.time()}.jpg',img_default)
        time.sleep(0.3)

    return[lf, fb, ud, yv]

while True:
    vals = getkeyboardInput()
    me.send_rc_control(vals[0],vals[1],vals[2],vals[3])
    time.sleep(0.05)
    img = me.get_frame_read().frame
    img_default = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", img_default)
    cv2.waitKey(1)



