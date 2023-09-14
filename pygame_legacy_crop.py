import pygame
import pygame.camera
import mediapipe as mp
import statistics
import cv2
import numpy as np
import math

import tensorflow as tf
from cameraman.model import (
    SoundModel,
)
from cameraman.utils import parse_audio_result

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
hand_model = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

model0 = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
model1 = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
import time
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python.audio.core import audio_record


pygame.init()

# default cavas size
gameDisplay = pygame.display.set_mode((2048, 1152), 0)

pygame.camera.init()
cameras = pygame.camera.list_cameras()

cam0 = pygame.camera.Camera(0, (1024, 576))
# cam1 = pygame.camera.Camera(1, (1024, 576))

cam0.start()
# cam1.start()

timestamp = 0
white = (255, 255, 255)

fonts = pygame.font.get_fonts()
font = pygame.font.SysFont("Arial", 15)


class EmotionModel:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(
            model_path="models/face_keypoint_classifier.tflite"
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.mapping = {0: "Laughing", 1: "Neutral", 2: "Angry"}

    def calc_landmarks(self, face_landmarker_result):
        point_coord = []
        # for point in face_landmarker_result.face_landmarks[0]:
        for point in face_landmarker_result.face_landmarks.landmark:
            point_coord.extend([point.x, point.y])
        return point_coord

    def inference(self, landmark_list):
        # if (
        #     face_landmarker_result is None
        #     or len(face_landmarker_result.face_landmarks) == 0
        # ):
        #     return None
        # landmark_list = self.calc_landmarks(face_landmarker_result)
        self.interpreter.set_tensor(
            self.input_details[0]["index"], np.array([landmark_list], dtype=np.float32)
        )

        self.interpreter.invoke()
        tflite_results = self.interpreter.get_tensor(self.output_details[0]["index"])

        inference_res = np.argmax(np.squeeze(tflite_results))

        return self.mapping[inference_res]


emotion = EmotionModel()


def convert_coord_drop(face_location):
    from_x, from_y, to_x, to_y = face_location
    margin = max(to_x - from_x, to_y - from_y)
    size = int(max(to_x - from_x, to_y - from_y) * 3)

    new_from_x = max(from_x - margin, 0)
    new_from_y = max(from_y - margin, 0)
    new_to_x = min(new_from_x + size, 1024)
    new_to_y = min(new_from_y + size, 576)

    return [new_from_x, new_from_y, new_to_x - new_from_x, new_to_y - new_from_y]


def convert_cropped_location(face_location):
    from_x, from_y, to_x, to_y = face_location
    margin = max(to_x - from_x, to_y - from_y)
    size = int(max(to_x - from_x, to_y - from_y) * 3)

    new_from_x = max(from_x - margin, 0)
    new_from_y = max(from_y - margin, 0)
    new_to_x = min(new_from_x + size, 1024)
    new_to_y = min(new_from_y + size, 576)
    return [
        new_from_x,
        new_from_y,
        new_to_x,
        new_to_y,
        new_to_x - new_from_x,
        new_to_y - new_from_y,
    ]


def hand_inference(image):
    results = hand_model.process(image)
    num_hands = 0

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)

    return num_hands


def face_inference(face_mesh_model, img):
    face_inference_results = []
    view = pygame.surfarray.array3d(img)
    view = view.transpose([1, 0, 2])
    image = view.copy()
    height, width, img_c = image.shape
    results = face_mesh_model.process(image)

    image.flags.writeable = True
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x_coordinates = [landmark.x for landmark in face_landmarks.landmark]
            y_coordinates = [landmark.y for landmark in face_landmarks.landmark]

            from_x = int(min(x_coordinates) * width)
            from_y = int(min(y_coordinates) * height)
            to_x = int(max(x_coordinates) * width)
            to_y = int(max(y_coordinates) * height)
            image = cv2.rectangle(image, (from_x, from_y), (to_x, to_y), (255, 0, 0), 2)

            (
                new_from_x,
                new_from_y,
                new_to_x,
                new_to_y,
                width,
                height,
            ) = convert_cropped_location((from_x, from_y, to_x, to_y))

            croped_image = image[new_from_y:new_to_y, new_from_x:new_to_x]
            num_hands = hand_inference(croped_image)

            lm_list = [[landmark.x, landmark.y] for landmark in face_landmarks.landmark]
            lm_list_flat = [item for sublist in lm_list for item in sublist]
            emotion_result = emotion.inference(lm_list_flat)

            face_inference_results.append(
                {
                    "face_locations_from_size": (new_from_x, new_from_y, width, height),
                    "face_emotion": emotion_result,
                    "num_hands": num_hands,
                }
            )
    image = image.transpose([1, 0, 2])
    return image, face_inference_results


while True:
    img0 = cam0.get_image()
    img0 = pygame.transform.flip(img0, True, False)
    #   audio_raw = record.read(15600)
    #     sound_result = sound_model.inference(audio_raw, timestamp)
    #     sound_result_str = parse_audio_result(sound_result)

    annotated_image_0, cropped_inference_result = face_inference(model0, img0)
    audio_result = open("audio_res.txt", "r").read()
    audio_inference = []

    annotated_image_0 = pygame.surfarray.make_surface(annotated_image_0)
    gameDisplay.blit(annotated_image_0, (0, 0))

    # annotated_image_1, direction_score_1 = face_inference(model1, img1)
    # annotated_image_1 = pygame.surfarray.make_surface(annotated_image_1)
    # gameDisplay.blit(annotated_image_1, (1024, 0))

    if len(cropped_inference_result) > 0:
        for idx, result in enumerate(cropped_inference_result):
            croped_surface = img0.subsurface(result["face_locations_from_size"])
            scaled_croped_surface = pygame.transform.scale(croped_surface, (576, 576))

            gameDisplay.blit(scaled_croped_surface, (0 + idx * 576, 576))
            gameDisplay.blit(
                font.render(f"DirectionScore: {result['face_emotion']}", True, white),
                (0 + idx * 576, 576),
            )
            gameDisplay.blit(
                font.render(f"Audio Classification: {audio_result}", True, white),
                (0 + idx * 576, 586),
            )
            gameDisplay.blit(
                font.render(f"Num Hands: {result['num_hands']}", True, white),
                (0 + idx * 576, 596),
            )

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cam0.stop()
            #            cam1.stop()
            pygame.quit()
            exit()
