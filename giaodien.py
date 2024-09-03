import sys
import cv2
from djitellopy import tello
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow,QWidget,QMessageBox

from gui import Ui_MainWindow
from thongsoke import Ui_widget
import time
import datetime
from math import *
qrc = cv2.QRCodeDetector()
import KeyPressModule as kp
kp.init()

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

fspeed  = 40       # toc do 15 cm/s
aspeed = 36       # toc do 50 do/s
fspeed_tello = 100
aspeed_tello = 100
T = 0.25
dT = fspeed * T
aT = aspeed * T
x,y = 150,750
a = 0
# bien to
yaw = 0
lr = 0
fb = 0
#bien toc do set cho drone
speed_lr = 0
speed_fb = 0
speed_yv = 0
go = 0   # đi tới hoặc quay về
co = 0
thoat = 0

hei = 0
i = 0
lan = 0
points = [(0,0),(150,750)]
start_point = (150,700)

#thong so kệ
cd = 600  #chieu dai ke
cr = 100  #chieu rong ke
kc = 100  #khoảng cách giữa 2 ke
n = 5   # số ke, mỗi ke có 2 dãy

# ke = 2
# cot = 2
cd_cot = 100
class MainWindow(QMainWindow):
    def __init__(self,index=0):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)

        self.uic.Btn_Connect.clicked.connect(self.start_capture_video)
        self.uic.Btn_Start.clicked.connect(self.start_capture_map)
        self.uic.Btn_nhapthongsokho.clicked.connect(self.open_form_thongso)

        self.thread = {}
    def open_form_thongso(self):
        self.form_thongso = Form_thongsoke()
        self.form_thongso.uic1.ln_cd.setText(str(cd))
        self.form_thongso.uic1.ln_cr.setText(str(cr))
        self.form_thongso.uic1.ln_kc.setText(str(kc))
        self.form_thongso.uic1.ln_soke.setText(str(n))
        self.form_thongso.show()


    def stop_capture_video(self):
        self.thread[1].stop()
    def start_capture_video(self):
        self.thread[1] = capture_video(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_wedcam)
    def start_capture_map(self):
        #me.takeoff()
        global vitri,ke,cot,hang,thoat
        thoat = 0
        vitri = self.uic.ln_vitri.text()
        tach = vitri.split('.')
        if len(tach) == 3:
            if tach[0].isdigit() and tach[1].isdigit() and tach[2].isdigit():
                ke = int(tach[0])
                cot = int(tach[1])
                hang = int(tach[2])
                # print(vitri)
                self.thread[2] = capture_map(index=2)
                self.thread[2].start()
                self.thread[2].signal_1.connect(self.show_map)
            else:
                QMessageBox.information(self,"Wrong","Nhập sai")
        else:
            QMessageBox.information(self, "Wrong", "Nhập thiếu")



    def show_map(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt_1(cv_img)
        self.uic.label.setPixmap(qt_img)
    def show_wedcam(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.uic.label_2.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        #print(h,w)
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(480,360, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    def convert_cv_qt_1(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        #print(h,w)
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(540,360, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class Form_thongsoke(QWidget):
    def __init__(self):
        super().__init__()
        self.uic1 = Ui_widget()
        self.uic1.setupUi(self)
        self.uic1.Btn_oke.clicked.connect(self.close_form)
        self.uic1.ln_cd.returnPressed.connect(self.move_cr)
        self.uic1.ln_cr.returnPressed.connect(self.move_kc)
        self.uic1.ln_kc.returnPressed.connect(self.move_soke)
        self.uic1.ln_soke.returnPressed.connect(self.move_oke)

    def move_cr(self):
        self.uic1.ln_cr.setFocus()
    def move_kc(self):
        self.uic1.ln_kc.setFocus()
    def move_soke(self):
        self.uic1.ln_soke.setFocus()
    def move_oke(self):
        self.uic1.Btn_oke.setFocus()
    def close_form(self):
        global cd, cr, kc, n
        cd = int(self.uic1.ln_cd.text())
        cr = int(self.uic1.ln_cr.text())
        kc = int(self.uic1.ln_kc.text())
        n = int(self.uic1.ln_soke.text())
        self.close()
class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super(capture_video, self).__init__()

    def run(self):
        global vitri,co
        #cap = cv2.VideoCapture(0)  # 'D:/8.Record video/My Video.mp4'
        while True:
            #_, cv_img = cap.read()
            cv_img = me.get_frame_read().frame
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            ret_qr, decode_inf, points, _ = qrc.detectAndDecodeMulti(cv_img)
            if ret_qr:
                for s, p in zip(decode_inf, points):
                    if s:
                        cv2.putText(cv_img, s, (p[0][0].astype(int), p[0][1].astype(int) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                                # print(p)
                        cv_img = cv2.polylines(cv_img, [p.astype(int)], True, (0, 255, 0), 8)
                        if co == 1:
                            global lan
                            lan = lan+1
                            current_time = datetime.datetime.now()
                            title = current_time.strftime("%Y-%m-%d-%Hh%M")
                            cv2.imwrite(f'Resources/Images/{vitri}/{title}_{lan}.jpg', cv_img)
                            time.sleep(0.3)
                            print('chup')
                            co = 0


                else:lan = 0
            cv_img = self.detect_orange_object(cv_img)
            self.signal.emit(cv_img)
    def stop(self):
        print("stop threading", self.index)
        self.terminate()

    def detect_orange_object(self,frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([5, 100, 100])
        upper = np.array([10, 255, 255])
        gray = cv2.inRange(hsv, lower, upper)
        cv2.imshow("gray", gray)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for ct in contours:
            area = cv2.contourArea(ct)
            if area > 2000:
                approx = cv2.approxPolyDP(ct, 0.01 * cv2.arcLength(ct, True), True)
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                x = approx.ravel()[0]
                y = approx.ravel()[1]

                cv2.putText(frame, 'orange', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
class capture_map(QThread):
    signal_1 = pyqtSignal(np.ndarray)
    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super(capture_map, self).__init__()

    def flytoPoint(self):
        global vitri,speed_lr,speed_fb,speed_yv,go,thoat
        global yaw, x, y, a, lr, fb, i,cot,ke,hang
        global co        # cờ báo đã tới vị trí cần đến và chụp hình
        #print(int(tach[0]))


        vt_cot = cot*100
        vt_hang = (hang-1)*100 + 50# chieu cao cua 1 hang là 100
        # hei = me.get_height()
        # print(hei)
        d = 0
        line = ke // 2
        du = ke % 2
        d_lr = (2 * line - 1) * 100

        if go == 0:
            # if hei > vt_hang:
            #     #me.move_up(10)
            #     time.sleep(0.5)
            if lr < d_lr:
                if lr == 0:
                    move_right = 1
                d = dT
                lr += d
                a = 0
            elif lr > d_lr:
                if lr == 0:
                    move_left = 1
                d = dT
                lr -= d
                a = -180
            elif fb < vt_cot:
                if fb == 0:
                    move_forward = 1
                fb += dT
                d = dT
                a = -90
            elif du == 1 and i < 10:
                if i == 0:
                    rotate_clockwise = 1
                i += 1
                yaw += 9
            elif du == 0 and i < 10:
                if i == 0:
                    rotate_counter_clockwise = 1
                i += 1
                yaw += -9
            else:
                co = 1
                go = 1
        elif go == 1:
            if du == 1 and i > 0:
                if i == 10:
                    rotate_counter_clockwise = 1
                i -= 1
                yaw -= 9
            elif du == 0 and i > 0:
                if i == 10:
                    rotate_clockwise = 1
                i -= 1
                yaw -= -9
            elif fb > 0:
                if fb >= vt_cot:
                    move_back = 1
                fb -= dT
                d = dT
                a = 90
            elif lr > 0:
                if lr >= d_lr:
                    move_left = 1
                d = dT
                lr -= d
                a = 180
            elif lr < 0:
                if lr <= d_lr:
                    move_right = 1
                d = dT
                lr += d
                a = 0
            else:
                thoat = 1
                go = 0
                points = [(0, 0), (150, 750)]

        time.sleep(T)
        # huong của drone
        x1 = int(x + 10 * cos(radians(yaw - 90)))
        y1 = int(y + 10 * sin(radians(yaw - 90)))
        a += yaw
        x += int(d * cos(radians(a)))
        y += int(d * sin(radians(a)))

        return [x, y, x1, y1, hang]

    def drawMap(self,img, start_point):
        start_y = start_point[1]
        for j in range(int(cd/cd_cot)):
            cv2.putText(img, f'{j + 1}', (25, start_y - int(cd_cot/2)), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 0), 3)
            start_y = start_y - cd_cot
        for i in range(n):
            start_x = start_point[0]
            start_y_2 = start_point[1]
            cv2.line(img, start_point, (start_point[0], start_point[1] - cd), (0, 0, 255), thickness=10)
            cv2.putText(img, f'{i * 2 + 1}', (start_point[0] - 50, start_point[1] - cd - 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 0), 3)
            cv2.putText(img, f'{i * 2 + 2}', (start_point[0] + 50, start_point[1] - cd - 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 0), 3)
            for h in range(int(cd/cd_cot)+1):
                cv2.circle(img, (start_x,start_y_2), 10, (255, 255, 255), cv2.FILLED)
                start_y_2 = start_y_2 - cd_cot
            start_point = (start_point[0] + cr + kc, start_point[1])




        return img

    def drawPoint(self,img, points, huong,hei):
        # for point in points:
        #     cv2.circle(img, point, 3, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, points[-1], 12, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, huong, 7, (255, 0, 0), cv2.FILLED)
        #cv2.putText(img, f'({(points[-1][0] - 150) / 100},{(points[-1][1] - 700) / 100})m',(points[-1][0] + 10, points[-1][1] + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(img,f'{int(hei)}',(points[-1][0] + 10, points[-1][1] + 30), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
        return img
    def run(self):
        global img,co
        while True:
            vals = self.flytoPoint()
            hei = vals[7]
            img = np.zeros((800, 1200, 3), np.uint8)
            huong = (vals[2], vals[3])
            img = self.drawMap(img, start_point)
            img = self.drawPoint(img, points, huong, hei)
            if points[-1][0] != vals[0] or points[-1][1] != vals[1]:
                points.append((vals[0], vals[1]))
            self.signal_1.emit(img)
            if points[-2][0] == 150 and points[-2][1] == 750:
                me.takeoff()
                #me.send_rc_control(0, 0, 5, 0)
                time.sleep(1)
            else:
                me.send_rc_control(int(vals[4]), int(vals[5]), 0, int(vals[6]))
            if co == 1:
                time.sleep(10)
            if thoat == 1:
                me.land()
                break

    def stop(self):
        print("stop threading", self.index)
        self.terminate()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())