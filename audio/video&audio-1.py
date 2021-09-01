import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os

import numpy as np
from PIL import ImageFont, ImageDraw, Image
import math

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

dB = 0

class VideoRecorder():

    # Video class based on openCV
    def __init__(self):

        self.open = True
        self.device_index = 0
        #self.fps = 6  # fps should be the minimum constant rate at which the camera can
        self.fourcc = "MJPG"  # capture images (with no decrease in speed over time; testing is required)
        self.frameSize = (640, 480)  # video formats and sizes also depend and vary according to the camera used
        #self.video_filename = "temp_video.avi"
        self.video_cap = cv2.VideoCapture(self.device_index)
        #self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        #self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
        #self.frame_counts = 1
        #self.start_time = time.time()

    # Video starts being recorded
    def record(self):

        # counter = 1
        #timer_start = time.time()
        #timer_current = 0

        while (self.open == True):
            ret, video_frame = self.video_cap.read()
            if (ret == True):
                # self.video_out.write(video_frame)
                # # print str(counter) + " " + str(self.frame_counts) + " frames written " + str(timer_current)
                # self.frame_counts += 1
                # # counter += 1
                # # timer_current = time.time() - timer_start
                #print(dB)

                if process_this_frame:
                    # Find all the faces and face encodings in the current frame of video
                    blob = cv2.dnn.blobFromImage(video_frame, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
                    face_mask_recognition_model.setInput(blob)
                    face_locations = face_mask_recognition_model.forward()
                    # print(face_locations)

                height, width = video_frame.shape[:2]
                result_image = video_frame.copy()
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

                    # if dB <= 40:
                    #     message = "조용함"
                    # elif dB == 60:
                    #     message = "수다 소리"
                    # else:
                    #     message = "욕설 감지됨"

                    if cnt == 1:
                        # print(age_list[age])
                        if age_list[age] == '(0 ~ 2)':
                            cv2.rectangle(result_image, pt1=(left, top), pt2=(right, bottom), thickness=2,
                                          color=colors[1],
                                          lineType=cv2.LINE_AA)
                            cv2.rectangle(result_image, pt1=(width - 115, 45), pt2=(width - 55, 90),
                                          color=(255, 255, 255), thickness=-1,
                                          lineType=cv2.LINE_AA)
                            cv2.putText(result_image, age_list[age], org=(left, top - 10),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.8, color=colors[1], thickness=2, lineType=cv2.LINE_AA)
                            # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None):
                            cv2.rectangle(result_image, pt1=(40, 45), pt2=(360, 90), color=(255, 255, 255),
                                          thickness=-1,
                                          lineType=cv2.LINE_AA)

                            fontpath = "font/gulim.ttc"
                            font = ImageFont.truetype(fontpath, 20)
                            img_pil = Image.fromarray(result_image)
                            draw = ImageDraw.Draw(img_pil)
                            # fill = rgb 색상
                            draw.text((50, 55), "위험! 아이가 혼자 차에 있습니다.", font=font, fill=(0, 0, 255, 128))
                            draw.text((width - 105, 55), "32°C", font=font, fill=(0, 255, 0, 128))
                            result_image = np.array(img_pil)

                        elif age_list[age] == '(4 ~ 6)':
                            cv2.rectangle(result_image, pt1=(left, top), pt2=(right, bottom), thickness=2,
                                          color=colors[1],
                                          lineType=cv2.LINE_AA)
                            cv2.rectangle(result_image, pt1=(width - 115, 45), pt2=(width - 55, 90),
                                          color=(255, 255, 255), thickness=-1,
                                          lineType=cv2.LINE_AA)
                            cv2.putText(result_image, age_list[age], org=(left, top - 10),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.8, color=colors[1], thickness=2, lineType=cv2.LINE_AA)
                            # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None):
                            cv2.rectangle(result_image, pt1=(40, 45), pt2=(360, 90), color=(255, 255, 255),
                                          thickness=-1,
                                          lineType=cv2.LINE_AA)

                            fontpath = "font/gulim.ttc"
                            font = ImageFont.truetype(fontpath, 20)
                            img_pil = Image.fromarray(result_image)
                            draw = ImageDraw.Draw(img_pil)
                            # fill = rgb 색상
                            draw.text((50, 55), "위험! 아이가 혼자 차에 있습니다.", font=font, fill=(0, 0, 255, 128))
                            draw.text((width - 105, 55), "32°C", font=font, fill=(0, 255, 0, 128))
                            result_image = np.array(img_pil)

                        else:
                            cv2.rectangle(result_image, pt1=(left, top), pt2=(right, bottom), thickness=2,
                                          color=colors[0],
                                          lineType=cv2.LINE_AA)
                            cv2.rectangle(result_image, pt1=(width - 115, 45), pt2=(width - 55, 90),
                                          color=(255, 255, 255), thickness=-1,
                                          lineType=cv2.LINE_AA)
                            cv2.putText(result_image, age_list[age], org=(left, top - 10),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.8, color=colors[0], thickness=2, lineType=cv2.LINE_AA)
                            fontpath = "font/gulim.ttc"
                            font = ImageFont.truetype(fontpath, 20)
                            img_pil = Image.fromarray(result_image)
                            draw = ImageDraw.Draw(img_pil)
                            # fill = rgb 색상
                            draw.text((width - 105, 55), "32°C", font=font, fill=(0, 255, 0, 128))
                            result_image = np.array(img_pil)

                    else:
                        cv2.rectangle(result_image, pt1=(left, top), pt2=(right, bottom), thickness=2, color=colors[0],
                                      lineType=cv2.LINE_AA)
                        cv2.rectangle(result_image, pt1=(width - 115, 45), pt2=(width - 55, 90), color=(255, 255, 255),
                                      thickness=-1,
                                      lineType=cv2.LINE_AA)
                        cv2.putText(result_image, age_list[age], org=(left, top - 10),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.8, color=colors[0], thickness=2, lineType=cv2.LINE_AA)
                        fontpath = "font/gulim.ttc"
                        font = ImageFont.truetype(fontpath, 20)
                        img_pil = Image.fromarray(result_image)
                        draw = ImageDraw.Draw(img_pil)
                        # fill = rgb 색상
                        draw.text((width - 105, 55), "32°C", font=font, fill=(0, 255, 0, 128))
                        result_image = np.array(img_pil)

                #self.video_out.write(result_image)
                # print str(counter) + " " + str(self.frame_counts) + " frames written " + str(timer_current)
                #self.frame_counts += 1

                # counter += 1

                # 프레임 조절하는 기능인듯
                # timer_current = time.time() - timer_start
                # time.sleep(0.16)

                cv2.imshow('video', result_image)
                cv2.waitKey(1)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Uncomment the following three lines to make the video to be
            # displayed to screen while recording

            #gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('video_frame', video_frame)
            # cv2.waitKey(1)
            # else:
            #     break

            # 0.16 delay -> 6 fps
            #

    # Finishes the video recording therefore the thread too
    # def stop(self):
    #
    #     if self.open == True:
    #
    #         self.open = False
    #         self.video_out.release()
    #         self.video_cap.release()
    #         cv2.destroyAllWindows()
    #
    #     else:
    #         pass

    # Launches the video recording function using a thread
    def start(self):
        video_thread = threading.Thread(target=self.record)
        video_thread.start()


class AudioRecorder():

    # Audio class based on pyAudio and Wave
    def __init__(self):
        global dB

        self.open = True
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 2
        self.format = pyaudio.paInt16
        # self.audio_filename = "temp_audio.wav"
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.frames_per_buffer)

    # Audio starts being recorded
    def record(self):
        global dB

        self.stream.start_stream()
        while (self.open == True):
            #data = self.stream.read(self.frames_per_buffer)
            #self.audio_frames.append(data)
            data = np.fromstring(self.stream.read(self.frames_per_buffer), dtype=np.int16)
            #print(type(data))
            a = int(math.log10(np.average(np.abs(data))))
            dB = 20 * a
            # print(dB)

            if self.open == False:
                break

    # Finishes the audio recording therefore the thread too
    def stop(self):

        if self.open == True:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()

        pass

    # Launches the audio recording function using a thread
    def start(self):
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()

def start_AVrecording():
    global video_thread
    global audio_thread

    audio_thread = AudioRecorder()
    video_thread = VideoRecorder()

    audio_thread.start()
    video_thread.start()

    return


# def start_video_recording(filename):
#     global video_thread
#
#     video_thread = VideoRecorder()
#     video_thread.start()
#
#     return filename
#
#
# def start_audio_recording(filename):
#     global audio_thread
#
#     audio_thread = AudioRecorder()
#     audio_thread.start()
#
#     return filename


if __name__ == "__main__":
    filename = "Default_user"
    #file_manager(filename)

    start_AVrecording()

    #time.sleep(10)

    #stop_AVrecording(filename)
    #print("Done")


