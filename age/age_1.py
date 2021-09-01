# 얼굴인식(마스크인식 카페 모델) + 연령 인식

# main.py
import cv2
import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# 영상 처리
def video_processing(video_path, background):
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    age_net = cv2.dnn.readNetFromCaffe(
        'C:/Users/user/Documents/GitHub/blackbox/models/deploy_age.prototxt',
        'C:/Users/user/Documents/GitHub/blackbox/models/age_net.caffemodel')

    face_mask_recognition_model = cv2.dnn.readNet(
        'C:/Users/user/Documents/GitHub/blackbox/models/face_mask_recognition.prototxt',
        'C:/Users/user/Documents/GitHub/blackbox/models/face_mask_recognition.caffemodel'
    )

    age_list = ['(0 ~ 2)', '(4 ~ 6)', '(8 ~ 12)', '(15 ~ 20)',
                '(25 ~ 32)', '(38 ~ 43)', '(48 ~ 53)', '(60 ~ 100)']

    #mask_detector_model = ai.create_model()

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    now_frame = 1

    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    out = None

    colors = [(0, 255, 0), (0, 0, 255)]
    #labels = ['with_mask', 'without_mask']

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
            if confidence < 0.5:
                continue

            left = int(face_locations[0, 0, i, 3] * width)
            top = int(face_locations[0, 0, i, 4] * height)
            right = int(face_locations[0, 0, i, 5] * width)
            bottom = int(face_locations[0, 0, i, 6] * height)

            #face_image = image[top:bottom, left:right]
            #face_image = cv2.resize(face_image, dsize=(224, 224))
            #face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            #predict = ai.predict(mask_detector_model, face_image)

            if right >= width or top >= right or left >= height or left >= bottom or right < 0 or left < 0 or bottom < 0 or top < 0:
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
                print(type(blob2))

            # Predict age
            age_net.setInput(blob2)
            age_preds = age_net.forward()
            age = age_preds.argmax()

            info = age_list[age]

            cv2.rectangle(
                result_image,
                pt1=(left, top),
                pt2=(right, bottom),
                thickness=2,
                color=colors[0],
                lineType=cv2.LINE_AA
            )
            # cv2.putText(result_image, info, (left, right - 15), 0, 0.5, (0, 255, 0), 1)
            cv2.putText(
                result_image,
                age_list[age],
                org=(left, top - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=colors[0],
                thickness=2,
                lineType=cv2.LINE_AA
            )

        if out is None:
            out = cv2.VideoWriter(
                'C:/Users/user/Documents/GitHub/blackbox/outputs/output.wmv',
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
    video_processing('C:/Users/user/Documents/GitHub/blackbox/data/baby1.mp4', False)
    #video_processing('C:/Users/user/Documents/GitHub/blackbox/data/04.mp4', False)

