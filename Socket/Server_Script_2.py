# -*- coding: utf-8 -*-
import io
import socket
import struct
import numpy as np
from PIL import Image

HOST = '192.168.0.203'
PORT = 8000
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    while True:
        conn, addr = s.accept()
        with conn:
            print('connected by', addr)
        while True:
            '''data = conn.recv(1024)
            if not data:
                break
                conn.sendall(data)'''
            image_len = struct.unpack('<L', conn.recv(1024))[0]
            if not image_len:
                break
            # Construct a stream to hold the image data and read the image
            # data from the connection
            image_stream = io.BytesIO()
            image_stream.write(conn.recv(image_len))

            image_stream.seek(0)

            image = Image.open(image_stream)

            # print('Image is %dx%d' % image.size)
            # image.verify()
            # print('Image is verified')

            # use numpy to convert the pil_image into a numpy array
            numpy_image = np.array(image)
            print(numpy_image)
