import io
import socket
import struct
import time
import picamera

# 접속하고 싶은 서버의 주소를 입력한다.
# ip주소를 직접 입력할 수 도 있고 hostname도 입력 가능하다.

# host = '127.0.0.1'
host = "localhost"

# 접속하고 싶은 포트를 입력한다.
port = 9123

#IPv4 체계, TCP 타입 소켓 객체를 생성
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 지정한 host와 prot를 통해 서버에 접속합니다.
client_socket.connect((host, port))

while 1:
    string = input("서버에 보내고 싶은 데이터를 입력하세요. : ")

    # 메시지 전송
    client_socket.sendall(string.encode())
    if string == "":
        break

    # 메시지 수신
    receive_data = client_socket.recv(1024)
    print("받은 데이터는 \"", receive_data.decode(), "\" 입니다.", sep="")
# 소켓을 닫는다.
client_socket.close()
print("접속을 종료합니다.")

try:
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        # Start a preview and let the camera warm up for 2 seconds
        camera.start_preview()
        time.sleep(2)

        # Note the start time and construct a stream to hold image data
        # temporarily (we could write it directly to connection but in this
        # case we want to find out the size of each capture first to keep
        # our protocol simple)
        start = time.time()
        stream = io.BytesIO()
        for foo in camera.capture_continuous(stream, 'jpeg'):
            # Write the length of the capture to the stream and flush to
            # ensure it actually gets sent
            # 메시지 전송
            client_socket.sendall(stream.encode())
            if string == "":
                break

            # connection.write(struct.pack('<L', stream.tell()))
            # connection.flush()
            # # Rewind the stream and send the image data over the wire
            # stream.seek(0)
            # connection.write(stream.read())
            # # If we've been capturing for more than 30 seconds, quit
            # if time.time() - start > 30:
            #     break
            # Reset the stream for the next capture
            stream.seek(0)
            stream.truncate()
    # Write a length of zero to the stream to signal we're done
    # connection.write(struct.pack('<L', 0))
finally:
    # connection.close()
    client_socket.close()
