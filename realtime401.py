import face_recognition
import cv2
import numpy as np
import os

from datetime import datetime

video_path = "./Video/test.mp4"
video_capture = cv2.VideoCapture(video_path)

def load_known_face_encodings(folder_path):
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

folder_path = "./known_people/"
known_face_encodings, known_face_names = load_known_face_encodings(folder_path)

print("已加载的人脸编码数量:", len(known_face_encodings))
print("已加载的人脸姓名:", known_face_names)

# Initialize
face_locations = []
process_this_frame = True

unknown_person_counter = 0

# Initialize variables for the last saved time
last_saved_time = None

# Initialize the folder path for saving unknown person images
unknown_person_folder = "./unknown_person_images/"

# Create the folder if it doesn't exist
if not os.path.exists(unknown_person_folder):
    os.makedirs(unknown_person_folder)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    #Break the loop if there is no more frame
    if not ret:
        break

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            current_time = datetime.now()
            
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                print(current_time, name, "returned home")
            else:
                # 如果上一次保存图片的时间和当前时间不在同一分钟内，则保存图片
                if last_saved_time is None or (current_time - last_saved_time).total_seconds() >= 60:
                    # 保存图像帧到本地
                    unknown_person_counter += 1
                    # 根据检测到的人脸位置裁剪人脸区域
                    for (top, right, bottom, left) in face_locations:
                        face_image = frame[top*4:bottom*4, left*4:right*4]
                        image_path = os.path.join(unknown_person_folder, f"{current_time}_unknown_person_{unknown_person_counter}.jpg")
                        cv2.imwrite(image_path, face_image)

                        # 更新上一次保存时间
                        last_saved_time = current_time

                    # 打印当前时间和提示信息
                    print(current_time, "Unknown person visit, image saved as:", image_path)
                            
            face_names.append(name)

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