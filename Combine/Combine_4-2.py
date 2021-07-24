# 얼굴 인식만 하는 코드

# main.py
import cv2
import os
import math
import face_recognition
from PIL import Image, ImageDraw

# 영상 처리
def video_processing(video_path, background):

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    now_frame = 1

    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    out = None

    colors = [(0, 255, 0), (0, 0, 255)] # 초록, 빨강
    #labels = ['with_mask', 'without_mask']

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        face_locations = face_recognition.face_locations(image)

        face_image = Image.fromarray(image)
        draw = ImageDraw.Draw(face_image)

        result_image = image.copy()

        for face_location in face_locations:
            top = face_location[0]
            right = face_location[1]
            bottom = face_location[2]
            left = face_location[3]

            cv2.rectangle(
                result_image,
                pt1=(left, top),
                pt2=(right, bottom),
                thickness=2,
                color=colors[0],
                lineType=cv2.LINE_AA
            )

        points = []

        #print(type(result_image))

        if out is None:
            out = cv2.VideoWriter(
                'C:/Users/user/Documents/GitHub/blackbox/outputs/output_test.wmv',
                fourcc,
                cap.get(cv2.CAP_PROP_FPS),
                # (image_frame.shape[1], image_frame.shape[0])
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
    #video_processing('C:/Users/user/PycharmProjects/OpenCV/doc/watershed_TestVideo/success_1.mp4', False)
    #video_processing('C:/Users/user/Documents/GitHub/blackbox/data/05.mp4', False)
    video_processing('C:/Users/user/Documents/GitHub/blackbox/data/baby2.mp4', False)
