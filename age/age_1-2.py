# 얼굴인식(마스크인식 카페 모델) + 연령 인식
# 상황 인식 기능 추가
#시연용 비디오 만든 영상

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
        print(height, width)

        blob = cv2.dnn.blobFromImage(image, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
        face_mask_recognition_model.setInput(blob)
        face_locations = face_mask_recognition_model.forward()

        result_image = image.copy()
        cnt = 0

        for i in range(face_locations.shape[2]):
            cnt += 1

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
                # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None):
                cv2.rectangle(result_image, pt1=(40, 105), pt2=(660, 165), color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

                fontpath = "font/gulim.ttc"
                font = ImageFont.truetype(fontpath, 40)
                img_pil = Image.fromarray(result_image)
                draw = ImageDraw.Draw(img_pil)
                # fill = rgb 색상
                draw.text((50, 115), "화면 안쪽으로 들어와주세요.", font=font, fill=(255, 0, 0, 128))
                result_image = np.array(img_pil)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(result_image, text, (50,100), font, 1, (255, 0, 0), 2)
            else:
                face = image[int(top):int(top + bottom - top), int(left):int(left + bottom - top)].copy()
                blob2 = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            #print(cnt)
            # Predict age
            age_net.setInput(blob2)
            age_preds = age_net.forward()
            age = age_preds.argmax()

            info = age_list[age]
            if cnt == 1:
                #print(age_list[age])
                if age_list[age] == '(0 ~ 2)':
                    cv2.rectangle(result_image, pt1=(left, top), pt2=(right, bottom), thickness=2, color=colors[1],
                                  lineType=cv2.LINE_AA)
                    #cv2.putText(result_image, age_list[age], org=(left, top - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                #fontScale=0.8, color=colors[1], thickness=2, lineType=cv2.LINE_AA)
                    # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None):
                    cv2.rectangle(result_image, pt1=(40, 45), pt2=(660, 105), color=(255, 255, 255), thickness=-1,
                                  lineType=cv2.LINE_AA)
                    cv2.rectangle(result_image, pt1=(width - 170, 45), pt2=(width - 55, 105), color=(255, 255, 255), thickness=-1,
                                 lineType=cv2.LINE_AA)

                    fontpath = "font/gulim.ttc"
                    font = ImageFont.truetype(fontpath, 40)
                    img_pil = Image.fromarray(result_image)
                    draw = ImageDraw.Draw(img_pil)
                    # fill = rgb 색상
                    draw.text((50, 55), "위험! 아이가 혼자 차에 있습니다.", font=font, fill=(0, 0, 255, 128))
                    draw.text((width - 160, 55), "32°C", font=font, fill=(0, 0, 255, 128))
                    result_image = np.array(img_pil)

                elif age_list[age] == '(4 ~ 6)':
                    cv2.rectangle(result_image, pt1=(left, top), pt2=(right, bottom), thickness=2, color=colors[1],
                                  lineType=cv2.LINE_AA)
                    #cv2.putText(result_image, age_list[age], org=(left, top - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                #fontScale=0.8, color=colors[1], thickness=2, lineType=cv2.LINE_AA)
                    # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None):
                    cv2.rectangle(result_image, pt1=(40, 45), pt2=(660, 105), color=(255, 255, 255), thickness=-1,
                                  lineType=cv2.LINE_AA)
                    cv2.rectangle(result_image, pt1=(width - 170, 45), pt2=(width - 55, 105), color=(255, 255, 255),
                                  thickness=-1, lineType=cv2.LINE_AA)

                    fontpath = "font/gulim.ttc"
                    font = ImageFont.truetype(fontpath, 40)
                    img_pil = Image.fromarray(result_image)
                    draw = ImageDraw.Draw(img_pil)
                    # fill = rgb 색상
                    draw.text((50, 55), "위험! 아이가 혼자 차에 있습니다.", font=font, fill=(0, 0, 255, 128))
                    draw.text((width - 160, 55), "32°C", font=font, fill=(0, 0, 255, 128))
                    result_image = np.array(img_pil)

                else:
                    cv2.rectangle(result_image, pt1=(left, top), pt2=(right, bottom), thickness=2, color=colors[0],
                                  lineType=cv2.LINE_AA)
                    #cv2.putText(result_image, age_list[age], org=(left, top - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                #fontScale=0.8, color=colors[0], thickness=2, lineType=cv2.LINE_AA)
                    cv2.rectangle(result_image, pt1=(width - 170, 45), pt2=(width - 55, 105), color=(255, 255, 255),
                                  thickness=-1, lineType=cv2.LINE_AA)

                    fontpath = "font/gulim.ttc"
                    font = ImageFont.truetype(fontpath, 40)
                    img_pil = Image.fromarray(result_image)
                    draw = ImageDraw.Draw(img_pil)
                    # fill = rgb 색상
                    draw.text((width - 160, 55), "32°C", font=font, fill=(0, 255, 0, 128))
                    result_image = np.array(img_pil)

            else:
                cv2.rectangle(result_image, pt1=(left, top), pt2=(right, bottom), thickness=2, color=colors[0],
                              lineType=cv2.LINE_AA)
                #cv2.putText(result_image, age_list[age], org=(left, top - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            #fontScale=0.8, color=colors[0], thickness=2, lineType=cv2.LINE_AA)
                cv2.rectangle(result_image, pt1=(width - 170, 45), pt2=(width - 55, 105), color=(255, 255, 255), thickness=-1,
                              lineType=cv2.LINE_AA)

                fontpath = "font/gulim.ttc"
                font = ImageFont.truetype(fontpath, 40)
                img_pil = Image.fromarray(result_image)
                draw = ImageDraw.Draw(img_pil)
                # fill = rgb 색상
                draw.text((width - 160, 55), "32°C", font=font, fill=(0, 255, 0, 128))
                result_image = np.array(img_pil)

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
    video_processing('C:/Users/user/Documents/GitHub/blackbox/data/test2-1.mp4', False)
    #video_processing('C:/Users/user/Documents/GitHub/blackbox/data/04.mp4', False)
