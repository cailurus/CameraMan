import mediapipe as mp
import cv2
import time
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from cameraman.model import GestureModel, FaceModel
from cameraman.utils import (
    put_text,
    draw_landmarks_on_image,
    draw_hand_landmarks_on_image,
)

WIDTH = 1280
HEIGHT = 720
FPS = 24.0

# 必须指定CAP_DSHOW(Direct Show)参数初始化摄像头,否则无法使用更高分辨率
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# 设置摄像头设备分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
# 设置摄像头设备帧率,如不指定,默认600
cap.set(cv2.CAP_PROP_FPS, FPS)


gesture_model = GestureModel()
face_model = FaceModel()


video = cv2.VideoCapture(0)

timestamp = 0

prev_frame_time, new_frame_time = 0, 0


while video.isOpened():
    ret, frame = video.read()

    if not ret:
        print("Ignoring empty frame")
        break

    timestamp += 1

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    hand_result = gesture_model.inference(mp_image, timestamp)
    face_res = face_model.inference(mp_image, timestamp)

    annotated_image = put_text(frame, fps, (0, 50))

    annotated_image = draw_hand_landmarks_on_image(annotated_image, hand_result)

    annotated_image = draw_landmarks_on_image(annotated_image, face_res)

    cv2.imshow("show", annotated_image)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        print("Closing Camera Stream")
        break

video.release()
cv2.destroyAllWindows()
