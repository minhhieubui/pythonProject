from djitellopy import Tello
from time import sleep

import cv2

tello = Tello()

tello.connect()

print(tello.get_battery())
tello.streamon()

while True:
    img = tello.get_frame_read().frame
    #img = cv2.resize(img, (360, 240))

    # Chuyển đổi hình ảnh về màu mặc định (BGR)
    img_default_color = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("image", img_default_color)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ hiển thị
tello.streamoff()
#cv2.release()
cv2.destroyAllWindows()