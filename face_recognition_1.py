import face_recognition
from PIL import Image, ImageDraw

face_image_path = 'C:/Users/user/PycharmProjects/OpenCV/People/people three.jpg'
#face_image_path = 'C:/Users/user/Documents/GitHub/blackbox/data/without_mask/6.jpg'

face_image_np = face_recognition.load_image_file(face_image_path)
face_locations = face_recognition.face_locations(face_image_np)

face_image = Image.fromarray(face_image_np)
draw = ImageDraw.Draw(face_image)

for face_location in face_locations:
    top = face_location[0]
    right = face_location[1]
    bottom = face_location[2]
    left = face_location[3]
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=4)

print(top, right, bottom, left)
#print(type(face_image), type(face_image_np))
#print(face_locations)

face_image.show()