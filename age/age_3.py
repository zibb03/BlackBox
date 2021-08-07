# 실시간 웹캠 처리 코드

import cv2
import pyaudio
import numpy as np

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_net = cv2.dnn.readNetFromCaffe(
    'C:/Users/user/Documents/GitHub/blackbox/models/deploy_age.prototxt',
    'C:/Users/user/Documents/GitHub/blackbox/models/age_net.caffemodel'
)
face_mask_recognition_model = cv2.dnn.readNet(
    'C:/Users/user/Documents/GitHub/blackbox/models/face_mask_recognition.prototxt',
    'C:/Users/user/Documents/GitHub/blackbox/models/face_mask_recognition.caffemodel'
)

age_list = ['(0 ~ 2)', '(4 ~ 6)', '(8 ~ 12)', '(15 ~ 20)',
            '(25 ~ 32)', '(38 ~ 43)', '(48 ~ 53)', '(60 ~ 100)']

video_capture = cv2.VideoCapture(0)

face_locations = []
process_this_frame = True

# CHUNK = 2 ** 10
# RATE = 44100
#
# p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
#                 frames_per_buffer=CHUNK, input_device_index=2)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
        face_mask_recognition_model.setInput(blob)
        face_locations = face_mask_recognition_model.forward()
        #print(face_locations)

    height, width = frame.shape[:2]
    result_image = frame.copy()

    # data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
    # print(int(np.average(np.abs(data))))

    for i in range(face_locations.shape[2]):
        confidence = face_locations[0, 0, i, 2]
        if confidence < 0.5:
            continue

        left = int(face_locations[0, 0, i, 3] * width)
        top = int(face_locations[0, 0, i, 4] * height)
        right = int(face_locations[0, 0, i, 5] * width)
        bottom = int(face_locations[0, 0, i, 6] * height)

        # face_image = rgb_small_frame[top:bottom, left:right]
        # face_image = cv2.resize(face_image, dsize=(224, 224))
        # face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        face = result_image[int(top):int(top + bottom - top), int(left):int(left + bottom - top)].copy()
        blob2 = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

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
            color=(0, 255, 0),
            lineType=cv2.LINE_AA
        )
        # cv2.putText(result_image, info, (left, right - 15), 0, 0.5, (0, 255, 0), 1)
        cv2.putText(
            result_image,
            age_list[age],
            org=(left, top - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )
    # stream.stop_stream()
    # stream.close()
    # p.terminate()


    # Display the resulting image
    #cv2.imshow('Video', frame)
    cv2.imshow('Video', result_image)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()