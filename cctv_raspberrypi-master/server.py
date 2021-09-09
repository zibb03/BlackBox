import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

import imagezmq

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
colors = [(0, 255, 0), (0, 0, 255)]

face_locations = []
process_this_frame = True

image_hub = imagezmq.ImageHub()
sender = imagezmq.ImageSender(connect_to='tcp://192.168.35.144:5555')

while True:
  rpi_name, frame = image_hub.recv_image()
  # print(type(image))

  # cv2.imshow(rpi_name, frame)
  # print(1)

  if cv2.waitKey(1) == ord('q'):
    break

  image_hub.send_reply(b'OK')
  #print(2)

  if process_this_frame:
    # Find all the faces and face encodings in the current frame of video
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1., size=(300, 300), mean=(104., 177., 123.)
    face_mask_recognition_model.setInput(blob)
    face_locations = face_mask_recognition_model.forward()
    # print(face_locations)

  height, width = frame.shape[:2]
  result_image = frame.copy()
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

    # face_image = rgb_small_frame[top:bottom, left:right]
    # face_image = cv2.resize(face_image, dsize=(224, 224))
    # face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

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
    age_net.setInput(blob2)
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

  # #display image to GUI
  # cv2.imshow(rpi_name, result_image)
  # cv2.waitKey(1)

  sender.send_image(rpi_name, result_image)
  #print(3)

