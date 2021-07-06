import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import dlib
# dlib를 상용 하는 이유는 XML파일을 이용하는 것 보다 dlib를 사용하여 인식 하는 것이 더 인식이 잘되는 듯 함
# dlib를 설치하기 위해서는 Anaconda에서 pip install cmake -> pip install dlib 설치


# dlib얼굴 인식 모델, 감정인식 모델 불러오기
# face_detection = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml') #사용 안함
detector = dlib.get_frontal_face_detector() # 얼굴 인식을 위한 dlib 기본 안면인식 모델
emotion_classifier = load_model('C:/Users/user/PycharmProjects/OpenCV/emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]

protoFile = "C:/Users/user/PycharmProjects/OpenCV/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "C:/Users/user/PycharmProjects/OpenCV/pose_iter_160000.caffemodel"
# 이미지 불러오기
frame = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/People/person3.jpg')
bgd = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/Background/Happy2.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
canvas = np.zeros((250, 300, 3), dtype="uint8")
dets = detector(frame, 1) # 얼굴인식 완료


# Perform emotion recognition only when face is detected
if len(dets) > 0:
    for k, d in enumerate(dets): # k = 얼굴 인식 갯수, d = 좌표(?)
        fX = d.left()
        fY = d.top()
        fW = d.right()
        fH = d.bottom()

        # print(fX, fY, fW, fH)

        roi = gray[fY:fH, fX:fW] # 얼굴 인식된 좌표
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # grabcut

        img = frame.copy()
        img2 = img.copy()

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        output2 = np.zeros(img.shape, np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # -------------------------------------------------------
        # MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
        BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                      "Background": 15}

        POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                      ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                      ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

        # 위의 path에 있는 network 불러오기
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

        # 이미지 읽어오기

        image = frame.copy()
        # frame.shape = 불러온 이미지에서 height, width, color 받아옴
        imageHeight, imageWidth, _ = image.shape

        # network에 넣기위해 전처리
        inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False,
                                        crop=False)

        # network에 넣어주기
        net.setInput(inpBlob)

        # 결과 받아오기
        output = net.forward()

        # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
        H = output.shape[2]
        W = output.shape[3]
        print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ", output.shape[3])  # 이미지 ID

        # 키포인트 검출시 이미지에 그려줌
        points = []
        for i in range(0, 15):
            # 해당 신체부위 신뢰도 얻음.
            probMap = output[0, i, :, :]

            # global 최대값 찾기
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # 원래 이미지에 맞게 점 위치 변경
            x = (imageWidth * point[0]) / W
            y = (imageHeight * point[1]) / H

            # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
            if prob > 0.1:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1,
                          lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                             lineType=cv2.LINE_AA)
                points.append((int(x), int(y)))
            else:
                points.append(None)

        # 이미지 복사
        imageCopy = image
        gx, gy, _ = imageCopy.shape
        rect2 = (10, 2, gx-15, gy-15)

        cv2.grabCut(imageCopy, mask, rect2, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)

        # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
        for pair in POSE_PAIRS:
            partA = pair[0]  # Head
            partA = BODY_PARTS[partA]  # 0
            partB = pair[1]  # Neck
            partB = BODY_PARTS[partB]  # 1


            # print(partA," 와 ", partB, " 연결\n")
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (255, 255, 255), 5)
                cv2.line(mask, points[partA], points[partB], 1, 15)

        # -------------------------------------------
        cv2.grabCut(imageCopy, mask, rect2, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        output2 = cv2.bitwise_and(imageCopy, imageCopy, mask=mask2)

        #--grabcut


        # Emotion predict
        #감정 인식하기
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        # Assign labeling
        cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (fX, fY), (fW, fH), (0, 0, 255), 2)


    # Label printing
    #사람이 여려명일 때는 마지막으로 감정을 인식한 사람의 감정 상태만 프린트함
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

# Open two windows
## Display image ("Emotion Recognition")
## Display probabilities of emotion


src = output2.copy()
y, x, _ = src.shape
bgd = cv2.resize(bgd, (x, y))
# cv2.imshow('bgd', bgd)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

aa = cv2.add(src, bgd, mask = binary)
result = cv2.add(aa, output2)
cv2.imshow('result', result)
cv2.imshow('output', output2)
cv2.imshow('Emotion Recognition', frame)
cv2.imshow("Probabilities", canvas)
cv2.waitKey(0)



# Clear program and close windows
cv2.destroyAllWindows()