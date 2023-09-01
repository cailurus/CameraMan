"""
    Copyright(c) 2021 the original author or authors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https: // www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
    or implied. See the License for the specific language governing
    permissions and limitations under the License.
 """

import cv2
import argparse
import time
from threading import Thread

from compreface import CompreFace
from compreface.service import RecognitionService

WIDTH = 1280
HEIGHT = 720
FPS = 24.0

cap1 = cv2.VideoCapture(0)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap1.set(cv2.CAP_PROP_FPS, FPS)

timestamp = 0
prev_frame_time = 0
new_frame_time = 0

compre_face: CompreFace = CompreFace(
    "http://10.172.207.55",
    "8000",
    {
        "limit": 0,
        "det_prob_threshold": 0.8,
        "prediction_count": 1,
        "face_plugins": "age,gender,mask",
        "status": False,
    },
)

recognition: RecognitionService = compre_face.init_face_recognition(
    "908ce6a9-5e09-466f-b4e2-eecdb8634cfa"
)


while True:
    # read and mirror
    ret, frame = cap1.read()
    # ret2, frame2 = cap2.read()

    frame = cv2.flip(frame, 1)

    if ret:
        # calc the FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        FPS = 1 / 30

        _, im_buf_arr = cv2.imencode(".jpg", frame)
        byte_im = im_buf_arr.tobytes()
        data = recognition.recognize(byte_im)
        # results = data.get("result")

        cv2.imshow("Camera Man Demo Main", frame)

    # if ret2:
    #     cv2.imshow("Sub", frame2)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        print("Closing Camera Stream")
        break

cap1.release()
# cap2.release()
cv2.destroyAllWindows()
