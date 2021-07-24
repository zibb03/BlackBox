# 각도 따라 스켈레톤 색 달라지는 코드 (시도중)

# main.py
import cv2
import os
import math
import face_recognition
from PIL import Image, ImageDraw

BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                      10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                      15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                      20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}

POSE_PAIRS_BODY_25 = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                      [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                      [11, 24], [22, 24], [23, 24]]

# 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
protoFile_body_25 = "C:/Users/user/Documents/GitHub/blackbox/models/pose_deploy.prototxt"

# 훈련된 모델의 weight 를 저장하는 caffemodel 파일
weightsFile_body_25 = "C:/Users/user/Documents/GitHub/blackbox/models/pose_iter_584000.caffemodel"

def output_keypoints(image, proto_file, weights_file, threshold, BODY_PARTS):
    global points

    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # 입력 이미지의 사이즈 정의
    image_height = 368
    image_width = 368

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False,
                                       crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()
    # The output is a 4D matrix :
    # The first dimension being the image ID ( in case you pass more than one image to the network ).
    # The second dimension indicates the index of a keypoint.
    # The model produces Confidence Maps and Part Affinity maps which are all concatenated.
    # For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps. Similarly, for MPI, it produces 44 points.
    # We will be using only the first few points which correspond to Keypoints.
    # The third dimension is the height of the output map.
    out_height = out.shape[2]
    # The fourth dimension is the width of the output map.
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = image.shape[:2]

    # 포인트 리스트 초기화
    points = []

    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정
        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)

        if prob > threshold:  # [pointed]
            cv2.circle(image, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            points.append((x, y))

        else:  # [not pointed]
            cv2.circle(image, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            points.append(None)

    return image


def output_keypoints_with_lines(POSE_PAIRS, frame):
    # 프레임 복사
    frame_line = frame.copy()

    # RElbow 과 RWrist 의 좌표값이 존재한다면
    if (points[3] is not None) and (points[4] is not None):
        calculate_degree1(point_1=points[3], point_2=points[4], frame=frame_line)

    # LElbow 과 LWrist 의 좌표값이 존재한다면
    if (points[5] is not None) and (points[6] is not None):
        calculate_degree2(point_1=points[5], point_2=points[6], frame=frame_line)

    for pair in POSE_PAIRS:
        part_a = pair[0]  # 0 (Head)
        part_b = pair[1]  # 1 (Neck)
        if points[part_a] and points[part_b]:
            print(f"[linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
            # Neck 과 MidHip 이라면 분홍색 선
            if part_a == 1 and part_b == 8:
                cv2.line(frame, points[part_a], points[part_b], (255, 0, 255), 3)
            else:  # 노란색 선
                cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 3)
        else:
            print(f"[not linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")

    return frame_line

def calculate_degree1(point_1, point_2, frame):
    # 역탄젠트 구하기
    dx = point_2[0] - point_1[0]
    dy = point_2[1] - point_1[1]
    rad = math.atan2(abs(dy), abs(dx))

    # radian 을 degree 로 변환
    deg = rad * 180 / math.pi

    # degree 가 45'보다 작으면 허리가 숙여졌다고 판단
    if deg < 45:
        # string = "Bend Down"
        # cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255))
        # print(f"[degree] {deg} ({string})")
        cv2.line(frame, points[3], points[4], (255, 0, 0), 3)
    # else:
    #     string = "Stand"
    #     cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255))
    #     print(f"[degree] {deg} ({string})")

def calculate_degree2(point_1, point_2, frame):
    # 역탄젠트 구하기
    dx = point_2[0] - point_1[0]
    dy = point_2[1] - point_1[1]
    rad = math.atan2(abs(dy), abs(dx))

    # radian 을 degree 로 변환
    deg = rad * 180 / math.pi

    # degree 가 45'보다 작으면 허리가 숙여졌다고 판단
    if deg < 45:
        # string = "Bend Down"
        # cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255))
        # print(f"[degree] {deg} ({string})")
        cv2.line(frame, points[5], points[6], (255, 0, 0), 3)
    # else:
    #     string = "Stand"
    #     cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255))
    #     print(f"[degree] {deg} ({string})")



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

        frame_man = output_keypoints(image=result_image, proto_file=protoFile_body_25, weights_file=weightsFile_body_25,
                                     threshold=0.1, BODY_PARTS=BODY_PARTS_BODY_25)
        image_frame = output_keypoints_with_lines(POSE_PAIRS=POSE_PAIRS_BODY_25, frame=frame_man)

        #print(type(result_image))

        if out is None:
            out = cv2.VideoWriter(
                'C:/Users/user/Documents/GitHub/blackbox/outputs/output test.wmv',
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
    video_processing('C:/Users/user/PycharmProjects/OpenCV/doc/watershed_TestVideo/success_1.mp4', False)
    #video_processing('C:/Users/user/Documents/GitHub/blackbox/data/05.mp4', False)
    #video_processing('C:/Users/user/Documents/GitHub/blackbox/data/model4.mp4', False)
