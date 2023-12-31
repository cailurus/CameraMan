import streamlit as st
import av
import mediapipe as mp
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from cameraman.utils import put_text, draw_landmarks_on_image, draw_hand, draw_face_box

from cameraman.model import GestureModel, FaceModel, FaceDetectorModel

gesture_model = GestureModel()
face_model = FaceModel()
face_detector_model = FaceDetectorModel()


st.title("LIVE")


timestamp = 0
prev_frame_time = 0
new_frame_time = 0


def callback(frame):
    # print(type(frame))
    frame = frame.to_ndarray(format="bgr24")
    print(frame.shape)

    global timestamp
    global prev_frame_time
    timestamp += 1

    new_frame_time = time.time()
    fps = f"FPS {1 / (new_frame_time - prev_frame_time):.2f}"
    prev_frame_time = new_frame_time

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    hand_result = gesture_model.inference(mp_image, timestamp)
    face_res = face_model.inference(mp_image, timestamp)

    annotated_image = put_text(frame, fps, (5, 50))
    annotated_image = draw_hand(annotated_image, hand_result, gesture_only=True)
    annotated_image = draw_landmarks_on_image(annotated_image, face_res)

    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=callback,
    media_stream_constraints={
        "video": {
            "width": 1080,
        }
    },
)
