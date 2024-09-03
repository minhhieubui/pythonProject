import cv2

qrc = cv2.QRCodeDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        ret_qr, decode_inf, points, _ = qrc.detectAndDecodeMulti(frame)
        if ret_qr :
            for s,p in zip(decode_inf,points):
                if s:
                    cv2.putText(frame,s,(p[0][0].astype(int),p[0][1].astype(int)-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3,cv2.LINE_AA)
                    #print(p)
                frame = cv2.polylines(frame,[p.astype(int)],True,(0,0,255),8)
        cv2.imshow('QR_code',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
