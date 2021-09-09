import io
import socket
import struct
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

host = "192.168.35.171"
port = 8000

# IPv4 체계, TCP 타입 소켓 객체를 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 포트를 사용 중 일때 에러를 해결하기 위한 구문
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# ip주소와 port번호를 함께 socket에 바인드 한다. #포트의 범위는 1-65535 사이의 숫자를 사용할 수 있다.
server_socket.bind((host, port))

# 서버가 클라이언트의 접속을 허용한다.
server_socket.listen()

# 클라이언트 함수가 접속하면 새로운 소켓을 반환한다.
client_socket, addr = server_socket.accept()
print("접속한 클라이언트의 주소 입니다. : ", addr)

while 1:
    image_len = struct.unpack('<L', server_socket.read(struct.calcsize('<L')))[0]
    image_len = struct.unpack('<L', server_socket.read(struct.calcsize('<L')))[0]
    if not image_len:
        break
    # Construct a stream to hold the image data and read the image
    # data from the connection
    image_stream = io.BytesIO()
    image_stream.write(server_socket.read(image_len))
    # Rewind the stream, open it as an image with PIL and do some
    # processing on it
    image_stream.seek(0)
    image = Image.open(image_stream)

    # print('Image is %dx%d' % image.size)
    # image.verify()
    # print('Image is verified')

    # use numpy to convert the pil_image into a numpy array
    numpy_image = np.array(image)
    print(numpy_image)

    # string = client_socket.recv(1024).decode()
    # if string == "":
    #     break
    # print("받은 데이터는 \"", string, "\" 입니다.", sep="")
    # client_socket.sendall(string.encode())
    # # 소켓을 닫는다.
    # print("접속을 종료합니다.")
    # client_socket.close()
    # server_socket.close()