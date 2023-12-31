import mediapipe as mp
import cv2
import time
from cameraman.model import (
    GestureModel,
    FaceModel,
    FaceDetectorModel,
    EmotionModel,
    SoundModel,
)
from cameraman.utils import (
    put_text,
    draw_face_landmarks,
    draw_hand,
    parse_audio_result,
    draw_face_direction,
    #    face_detector_visualize,
)

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
emotion_model = EmotionModel()

sound_model = SoundModel()

timestamp = 0
prev_frame_time = 0
new_frame_time = 0

from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python.audio.core import audio_record

buffer_size, sample_rate, num_channels = 15600, 44100, 1
audio_format = containers.AudioDataFormat(num_channels, sample_rate)
record = audio_record.AudioRecord(num_channels, sample_rate, buffer_size)
audio_data = containers.AudioData(buffer_size, audio_format)

input_length_in_second = (
    float(len(audio_data.buffer)) / audio_data.audio_format.sample_rate
)

interval_between_inference = input_length_in_second * (1 - 0.5)
pause_time = interval_between_inference * 0.1

last_inference_time = time.time()


from cameraman.model import BUFFER_SIZE, record

record.start_recording()

while True:
    # read and mirror
    ret, frame = cap1.read()
    # ret2, frame2 = cap2.read()

    frame = cv2.flip(frame, 1)

    if ret:
        timestamp += 1

        # calc the FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # face_detector_res = face_detector_model.inference(mp_image, timestamp)
        # locations = face_detector_visualize(frame, face_detector_res)

        hand_result = gesture_model.inference(mp_image, timestamp)
        # face_keypoints_result = face_model.inference(mp_image, timestamp)
        # emotion_result = emotion_model.inference(face_keypoints_result)

        # annotated_image = put_text(frame, fps, (5, 50))
        # annotated_image = draw_hand(annotated_image, hand_result, gesture_only=True)
        # annotated_image = put_text(frame, emotion_result, (5, 150))
        # annotated_image, direction_score = draw_face_direction(
        #     annotated_image, face_keypoints_result
        # )

        # annotated_image = draw_face_landmarks(annotated_image, face_keypoints_result)

        audio_raw = record.read(BUFFER_SIZE)

        sound_result = sound_model.inference(audio_raw, timestamp)
        sound_result_str = parse_audio_result(sound_result)
        print(sound_result_str)
        # annotated_image = put_text(annotated_image, sound_result_str, (5, 150))

        # cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # cv2.imshow("Camera Man Demo Main", annotated_image)

    # if ret2:
    #     cv2.imshow("Sub", frame2)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        print("Closing Camera Stream")
        break

cap1.release()
# cap2.release()
cv2.destroyAllWindows()
