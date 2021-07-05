# main.py
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import dlib
import random

import cv2
import os
import ai

emotion_classifier = load_model('C:/Users/user/PycharmProjects/OpenCV/emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

protoFile = "C:/Users/user/PycharmProjects/OpenCV/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "C:/Users/user/PycharmProjects/OpenCV/pose_iter_160000.caffemodel"

# 영상 처리
def video_processing(video_path, background):
    face_mask_recognition_model = cv2.dnn.readNet(
        'models/face_mask_recognition.prototxt',
        'models/face_mask_recognition.caffemodel'
    )

    mask_detector_model = ai.create_model()

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    now_frame = 1

    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    out = None

    colors = [(0, 255, 0), (0, 0, 255)]
    labels = ['with_mask', 'without_mask']

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        height, width = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
        face_mask_recognition_model.setInput(blob)
        face_locations = face_mask_recognition_model.forward()

        result_image = image.copy()

        for i in range(face_locations.shape[2]):
            confidence = face_locations[0, 0, i, 2]

            left = int(face_locations[0, 0, i, 3] * width)
            top = int(face_locations[0, 0, i, 4] * height)
            right = int(face_locations[0, 0, i, 5] * width)
            bottom = int(face_locations[0, 0, i, 6] * height)


        for i in range(face_locations.shape[2]):
            confidence = face_locations[0, 0, i, 2]
            if confidence < 0.5:
                continue

            left = int(face_locations[0, 0, i, 3] * width)
            top = int(face_locations[0, 0, i, 4] * height)
            right = int(face_locations[0, 0, i, 5] * width)
            bottom = int(face_locations[0, 0, i, 6] * height)

            fX = left
            fY = top
            fW = width
            fH = height

            face_image = image[top:bottom, left:right]
            face_image = cv2.resize(face_image, dsize=(224, 224))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            predict = ai.predict(mask_detector_model, face_image)

            gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

            # roi = gray[fY:fH, fX:fW] #얼굴 인식된 좌표
            roi = gray[fY:fH, fX:fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Emotion predict
            # 감정 인식하기
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            cv2.putText(result_image, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(result_image, (fX, fY), (fW, fH), (0, 0, 255), 2)

            '''
            cv2.rectangle(
                result_image,
                pt1=(left, top),
                pt2=(right, bottom),
                thickness=2,
                color=colors[predict],
                lineType=cv2.LINE_AA
            )

            cv2.putText(
                result_image,
                text=labels[predict],
                org=(left, top - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=colors[predict],
                thickness=2,
                lineType=cv2.LINE_AA
            )
            '''
            '''
            img = result_image.copy()
            img2 = img.copy()

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            output2 = np.zeros(img.shape, np.uint8)

            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
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
            image = result_image.copy()
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
                    cv2.circle(result_image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1,
                               lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
                    cv2.putText(result_image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                lineType=cv2.LINE_AA)
                    points.append((int(x), int(y)))
                else:
                    points.append(None)

            # 이미지 복사
            imageCopy = image
            gx, gy, _ = imageCopy.shape
            rect2 = (10, 2, gx - 15, gy - 15)

            cv2.grabCut(imageCopy, mask, rect2, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)

            # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
            for pair in POSE_PAIRS:
                partA = pair[0]  # Head
                partA = BODY_PARTS[partA]  # 0
                partB = pair[1]  # Neck
                partB = BODY_PARTS[partB]  # 1

                # print(partA," 와 ", partB, " 연결\n")
                if points[partA] and points[partB]:
                    cv2.line(result_image, points[partA], points[partB], (255, 255, 255), 5)
                    cv2.line(mask, points[partA], points[partB], 1, 15)
'''
        if out is None:
            out = cv2.VideoWriter(
                #'outputs/output.wmv',
                'outputs/output1.wmv',
                fourcc,
                cap.get(cv2.CAP_PROP_FPS),
                (image.shape[1], image.shape[0])
            )
        else:
            out.write(result_image)

        # (10/400): 11%
        print('(' + str(now_frame) + '/' + str(frame_count) + '): ' + str(now_frame * 100 // frame_count) + '%')
        now_frame += 1

        if not background:
            cv2.imshow('result', result_image)
            if cv2.waitKey(1) == ord('q'):
                break

    out.release()
    cap.release()

if __name__ == '__main__':
    video_processing('data/04.mp4', False)