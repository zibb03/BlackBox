#오디오 불러와 소리 크기 출력

import pyaudio
import wave
import numpy as np

CHUNK = 1024
RATE = 44100

path = 'C:/Users/user/Documents/GitHub/blackbox/data/baby3.wav'

with wave.open(path, 'rb') as f:
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)

    # stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
    #                 frames_per_buffer=CHUNK, input_device_index=2)

    data = f.readframes(CHUNK)

    while(True):
        #stream.write(data)
        #data = f.readframes(CHUNK)
        data = np.fromstring(f.readframes(CHUNK), dtype=np.int16)
        #data = np.fromBuffer(f.readframes(CHUNK), dtype=np.int16)
        print(int(np.average(np.abs(data))))

    stream.stop_stream()
    stream.close()

    p.terminate()




