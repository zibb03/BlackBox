# -*- coding: utf-8 -*-
import socket
HOST = '192.168.0.203'
PORT = 8000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    while True:
        conn, addr = s.accept()
        with conn: print('connected by', addr)
        while True:
            data = conn.recv(1024)
            if not data:
                break
                conn.sendall(data)
