import socket
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
