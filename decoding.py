import numpy as np
import cv2
import time
import base64

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    time.sleep(0.1)
    ret, frame = cap.read()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    b64data = base64.b64encode(frame)
    #위까지 base64로 인코딩  아래부터는 디코딩
    frame = base64.b64decode(b64data)
    froms = np.frombuffer(frame, dtype='uint8')
    frame = cv2.imdecode(froms,cv2.IMREAD_COLOR)
    cv2.imshow("test", frame)
    if cv2.waitKey(1) > 0: break
cv2.destroyAllWindows()