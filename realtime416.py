import face_recognition
import cv2
import numpy as np
import os
import re

from datetime import datetime

def get_starting_unknown_id(unknown_path):
    max_id = 0
    for filename in os.listdir(unknown_path):
        match = re.search(r'Unknown_(\d+).jpg', filename)
        if match:
            id = int(match.group(1))
            if id > max_id:
                max_id = id
    return max_id + 1

def load_face_encodings(folder_path):
    face_encodings = []
    face_names = []

    # 遍历文件夹中的每张图片
    for filename in os.listdir(folder_path):
        # 仅处理扩展名为jpg、jpeg、png的文件
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # 加载图片
            image_path = os.path.join(folder_path, filename)
            face_image = face_recognition.load_image_file(image_path)

            # 生成面部编码
            face_encoding = face_recognition.face_encodings(face_image)[0]

            # 获取人物姓名（假设文件名即为姓名）
            name = os.path.splitext(filename)[0]

            # 添加到列表中
            face_encodings.append(face_encoding)
            face_names.append(name)

    return face_encodings, face_names


video_path = "./Video/test.mp4"
video_capture = cv2.VideoCapture(video_path)
known_path = "./known_people/"
unknown_path = "./unknown_people/"
known_face_encodings, known_face_names = load_face_encodings(known_path)
unknown_face_encodings, unknown_face_names = load_face_encodings(unknown_path)

all_face_encodings = known_face_encodings + unknown_face_encodings
all_face_names = known_face_names + unknown_face_names

print("已加载的人脸编码数量:", len(all_face_encodings))
print("已加载的人脸姓名:", all_face_names)

# Initialize
face_locations = []
process_this_frame = True
unknown_person_counter = get_starting_unknown_id(unknown_path)
last_saved_time = None

# Initialize the folder path for saving unknown person images
unknown_person_folder = "./unknown_person_images/"
if not os.path.exists(unknown_person_folder):
    os.makedirs(unknown_person_folder)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Only process every other frame of video to save time
    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            current_time = datetime.now()
            
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(all_face_encodings, face_encoding)
            name = "moshengren"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(all_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = all_face_names[best_match_index]
                print(current_time, name, "returned home")
            else:
                if "moshengren" in name:
                    unknown_person_counter += 1
                    for (top, right, bottom, left) in face_locations:
                        face_image = frame[top*4:bottom*4, left*4:right*4]
                        image_path = os.path.join(unknown_path, f"Unknown_{unknown_person_counter}.jpg")
                        cv2.imwrite(image_path, face_image)
                    # Add the unknown face encoding and name to the known lists
                    all_face_encodings.append(face_encoding)
                    all_face_names.append(f"Unknown_{unknown_person_counter}")
                            
            face_names.append(name)
            print(face_names)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()