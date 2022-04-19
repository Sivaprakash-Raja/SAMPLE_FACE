import face_recognition
import cv2
import os

def read_img(path):
    img = cv2.imread(path)
    print(path)
    (h, w) = img.shape[:2]
    width = 500
    ratio = width / float(w)
    height = int(h * ratio)
    #cv2.imshow(img)
            
    return cv2.resize(img, (width, height))


data = read_img("C:\\Users\\shiva\\OneDrive\\Desktop\\face_detection\\known\\dhoni.jfif")
print(data)


known_encodings = []
known_names = []
known_dir = "C:\\Users\\shiva\\OneDrive\\Desktop\\face_detection\\known"

for file in os.listdir(known_dir):
    img = read_img( known_dir + '\\' + file)
    
    img_enc = face_recognition.face_encodings(img)[0]
    known_encodings.append(img_enc)
    known_names.append(file.split('.')[0])


unknown_dir = 'C:\\Users\\shiva\\OneDrive\\Desktop\\face_detection\\unknown'

for file in os.listdir(unknown_dir):
    print("Processing", file)
    img = read_img(unknown_dir + '\\' + file)
    img_enc = face_recognition.face_encodings(img)[0]

    results = face_recognition.compare_faces(known_encodings, img_enc)
    # print(face_recognition.face_distance(known_encodings, img_enc))

    for i in range(len(results)):
        if results[i]:
            name = known_names[i]
            (top, right, bottom, left) = face_recognition.face_locations(img)[0]
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(img, name, (left+2, bottom+20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            cv2.imshow(img)
            

