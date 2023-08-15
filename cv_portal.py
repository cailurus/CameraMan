import mediapipe as mp
import cv2
import time
from cameraman.model import GestureModel, FaceModel, FaceDetectorModel
from cameraman.utils import put_text, draw_face_landmarks, draw_hand

WIDTH = 1280
HEIGHT = 720
FPS = 24.0

cap1 = cv2.VideoCapture(0)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap1.set(cv2.CAP_PROP_FPS, FPS)

cap2 = cv2.VideoCapture(1)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap2.set(cv2.CAP_PROP_FPS, FPS)


gesture_model = GestureModel()
face_model = FaceModel()
face_detector_model = FaceDetectorModel()

timestamp = 0
prev_frame_time = 0
new_frame_time = 0

while True:
    # read and mirror
    ret, frame = cap1.read()
    ret2, frame2 = cap2.read()

    frame = cv2.flip(frame, 1)

    if ret:
        timestamp += 1

        # calc the FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        hand_result = gesture_model.inference(mp_image, timestamp)
        face_result = face_model.inference(mp_image, timestamp)

        annotated_image = put_text(frame, fps, (5, 50))

        annotated_image = draw_hand(annotated_image, hand_result, gesture_only=True)

        annotated_image = draw_face_landmarks(annotated_image, face_result)

        cv2.imshow("Camera Man Demo Main", annotated_image)

    if ret2:
        cv2.imshow("Sub", frame2)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        print("Closing Camera Stream")
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
