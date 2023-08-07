import mediapipe as mp
import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from cameraman.model import GestureModel
from cameraman.utils import draw_gesture


model = GestureModel()


video = cv2.VideoCapture(0)

timestamp = 0
while video.isOpened():
    ret, frame = video.read()

    if not ret:
        print("Ignoring empty frame")
        break

    timestamp += 1
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    fps = video.get(cv2.CAP_PROP_FPS)

    rr = model.inference(mp_image, timestamp)

    annotated_image = draw_gesture(frame, f"{rr}")

    cv2.imshow("show", annotated_image)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        print("Closing Camera Stream")
        break

video.release()
cv2.destroyAllWindows()
