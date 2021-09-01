# 실시간 웹캠 처리 코드

import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_net = cv2.dnn.readNetFromCaffe(
    'C:/Users/user/Documents/GitHub/blackbox/models/deploy_age.prototxt',
    'C:/Users/user/Documents/GitHub/blackbox/models/age_net.caffemodel'
)
# face_mask_recognition_model = cv2.dnn.readNet(
#     'C:/Users/user/Documents/GitHub/blackbox/models/face_mask_recognition.prototxt',
#     'C:/Users/user/Documents/GitHub/blackbox/models/face_mask_recognition.caffemodel'
# )
cascade_filename = 'C:/Users/user/Documents/GitHub/blackbox/models/haarcascade_frontalface_alt.xml'
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

age_list = ['(0 ~ 2)', '(4 ~ 6)', '(8 ~ 12)', '(15 ~ 20)',
            '(25 ~ 32)', '(38 ~ 43)', '(48 ~ 53)', '(60 ~ 100)']
colors = [(0, 255, 0), (0, 0, 255)]

video_capture = cv2.VideoCapture(0)

face_locations = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    # if process_this_frame:
    #     # Find all the faces and face encodings in the current frame of video
    #     blob = cv2.dnn.blobFromImage(frame, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
    #     face_mask_recognition_model.setInput(blob)
    #     face_locations = face_mask_recognition_model.forward()
    #     #print(face_locations)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = cascade.detectMultiScale(gray,  # 입력 이미지
                                       scaleFactor=1.1,  # 이미지 피라미드 스케일 factor
                                       minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                       minSize=(20, 20)  # 탐지 객체 최소 크기
                                       )

    height, width = frame.shape[:2]
    result_image = frame.copy()
    cnt = 0

    for box in results:
        x, y, w, h = box
        face = frame[int(y):int(y + h), int(x):int(x + h)].copy()
        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        left = int(x)
        top = int(y)
        right = int(x + w)
        bottom = int(y + h)

        if right >= width or top >= right or left >= height or left >= bottom or right < 0 or left < 0 or bottom < 0 or top < 0:
            # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None):
            cv2.rectangle(result_image, pt1=(40, 90), pt2=(360, 130), color=(255, 255, 255), thickness=-1,
                          lineType=cv2.LINE_AA)

            fontpath = "font/gulim.ttc"
            font = ImageFont.truetype(fontpath, 20)
            img_pil = Image.fromarray(result_image)
            draw = ImageDraw.Draw(img_pil)
            # fill = rgb 색상
            draw.text((50, 100), "화면 안쪽으로 들어와주세요.", font=font, fill=(255, 0, 0, 128))
            result_image = np.array(img_pil)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(result_image, text, (50,100), font, 1, (255, 0, 0), 2)
        else:
            face = result_image[int(top):int(top + bottom - top), int(left):int(left + bottom - top)].copy()
            blob2 = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_preds.argmax()

        info = age_list[age]

        if cnt == 1:
            # print(age_list[age])
            if age_list[age] == '(0 ~ 2)':
                cv2.rectangle(result_image, pt1=(left, top), pt2=(right, bottom), thickness=2, color=colors[1],
                              lineType=cv2.LINE_AA)
                cv2.putText(result_image, age_list[age], org=(left, top - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8, color=colors[1], thickness=2, lineType=cv2.LINE_AA)
                # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None):
                cv2.rectangle(result_image, pt1=(40, 45), pt2=(360, 90), color=(255, 255, 255), thickness=-1,
                              lineType=cv2.LINE_AA)

                fontpath = "font/gulim.ttc"
                font = ImageFont.truetype(fontpath, 20)
                img_pil = Image.fromarray(result_image)
                draw = ImageDraw.Draw(img_pil)
                # fill = rgb 색상
                draw.text((50, 55), "위험! 아이가 혼자 차에 있습니다.", font=font, fill=(0, 0, 255, 128))
                result_image = np.array(img_pil)

            elif age_list[age] == '(4 ~ 6)':
                cv2.rectangle(result_image, pt1=(left, top), pt2=(right, bottom), thickness=2, color=colors[1],
                              lineType=cv2.LINE_AA)
                cv2.putText(result_image, age_list[age], org=(left, top - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8, color=colors[1], thickness=2, lineType=cv2.LINE_AA)
                # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None):
                cv2.rectangle(result_image, pt1=(40, 45), pt2=(360, 90), color=(255, 255, 255), thickness=-1,
                              lineType=cv2.LINE_AA)

                fontpath = "font/gulim.ttc"
                font = ImageFont.truetype(fontpath, 20)
                img_pil = Image.fromarray(result_image)
                draw = ImageDraw.Draw(img_pil)
                # fill = rgb 색상
                draw.text((50, 55), "위험! 아이가 혼자 차에 있습니다.", font=font, fill=(0, 0, 255, 128))
                result_image = np.array(img_pil)

            else:
                cv2.rectangle(result_image, pt1=(left, top), pt2=(right, bottom), thickness=2, color=colors[0],
                              lineType=cv2.LINE_AA)
                cv2.putText(result_image, age_list[age], org=(left, top - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8, color=colors[0], thickness=2, lineType=cv2.LINE_AA)

        else:
            cv2.rectangle(result_image, pt1=(left, top), pt2=(right, bottom), thickness=2, color=colors[0],
                          lineType=cv2.LINE_AA)
            cv2.putText(result_image, age_list[age], org=(left, top - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8, color=colors[0], thickness=2, lineType=cv2.LINE_AA)

    # Display the resulting image
    # cv2.imshow('Video', frame)
    cv2.imshow('Video', result_image)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()