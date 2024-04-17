import cv2

# 加载Haar级联分类器
face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

# 打开摄像头
video_path = "./Video/ex1.mp4"
cap = cv2.VideoCapture(video_path)

# 读取第一帧
ret, frame = cap.read()

# 设定初始的跟踪窗口
faces = face_cascade.detectMultiScale(frame, 1.3, 5)
while len(faces) == 0:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
face_x, face_y, w, h = faces[0]
track_window = (face_x, face_y, w, h)

# 创建KCF跟踪器
tracker = cv2.TrackerKCF_create()
init_track = tracker.init(frame, track_window)

while True:
    ret, frame = cap.read()
    if ret:
        # 更新跟踪器的位置
        success, box = tracker.update(frame)
        if success:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        cv2.imshow('Tracking', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()