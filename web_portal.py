import streamlit as st
import cv2
import av
import mediapipe as mp
import time
from streamlit_webrtc import webrtc_streamer
from cameraman.utils import (
    put_text,
    draw_landmarks_on_image,
    draw_hand_landmarks_on_image,
)

from cameraman.model import GestureModel, FaceModel

gesture_model = GestureModel()
face_model = FaceModel()


st.title("My first Streamlit app")
st.write("Hello, world")


timestamp = 0
prev_frame_time = 0
new_frame_time = 0


def callback(frame):
    global timestamp
    global prev_frame_time
    timestamp += 1

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    frame = frame.to_ndarray(format="bgr24")
    #
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    hand_result = gesture_model.inference(mp_image, timestamp)
    face_res = face_model.inference(mp_image, timestamp)
    #
    annotated_image = put_text(frame, fps, (0, 50))
    annotated_image = draw_hand_landmarks_on_image(annotated_image, hand_result)
    annotated_image = draw_landmarks_on_image(annotated_image, face_res)

    # return annotated_image

    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


webrtc_streamer(
    key="example",
    video_frame_callback=callback,
    media_stream_constraints={
        "video": {
            "width": 1920,
        }
    },
)
