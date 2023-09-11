import math

import mediapipe as mp
import numpy as np
import pygame
import pygame.camera

from cameraman.model import FaceDetectorModel, FaceModel, GestureModel, SoundModel
from cameraman.utils import (
    draw_face_direction,
    draw_face_landmarks,
    draw_hand,
    get_boundingbox_detector,
    parse_audio_result,
    put_text,
)

# gesture_model = GestureModel()
face_model_0 = FaceModel(num_faces=2)
# face_model_1 = FaceModel()

face_dedector_model = FaceDetectorModel()
# face_detector_model = FaceDetectorModel()
pygame.init()

# default cavas size
gameDisplay = pygame.display.set_mode((2048, 1152), 0)

pygame.camera.init()
cameras = pygame.camera.list_cameras()

cam0 = pygame.camera.Camera(0, (1024, 576))
cam1 = pygame.camera.Camera(1, (1024, 576))

cam0.start()
cam1.start()

timestamp = 0
white = (255, 255, 255)

fonts = pygame.font.get_fonts()
font = pygame.font.SysFont("Arial", 15)


def face_inference(face_model, img, timestamp):
    imgdata = pygame.surfarray.array3d(img)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgdata)
    face_keypoints_result = face_model.inference(mp_image, timestamp)
    annotated_image, face_locations = draw_face_landmarks(
        imgdata, face_keypoints_result
    )
    #    _, score = draw_face_direction(annotated_image, face_keypoints_result)
    score = 0

    annotated_image = pygame.surfarray.make_surface(annotated_image)

    return annotated_image, score, face_locations


import cv2


def face_detection_inference(face_detector_model, img, timestamp):
    imgdata = pygame.surfarray.array3d(img)
    imgdata = np.transpose(imgdata, (1, 0, 2))
    face_detector_res = face_detector_model.inference(imgdata)
    locations = get_boundingbox_detector(imgdata, face_detector_res)
    return locations


def convert_coord_draw_rect(face_location):
    from_x, from_y, to_x, to_y = face_location
    return (from_x, from_y, to_x - from_x, to_y - from_y)


SMOOTHING_SIZE = 80


def convert_coord_drop(face_location):
    from_x, from_y, to_x, to_y = face_location
    print(from_x, from_y, to_x, to_y)
    from_x = int(from_x / SMOOTHING_SIZE) * SMOOTHING_SIZE
    from_y = int(from_y / SMOOTHING_SIZE) * SMOOTHING_SIZE
    to_x = math.ceil(to_x / SMOOTHING_SIZE) * SMOOTHING_SIZE
    to_y = math.ceil(to_y / SMOOTHING_SIZE) * SMOOTHING_SIZE
    print(from_x, from_y, to_x, to_y)

    margin = max(to_x - from_x, to_y - from_y) // 2
    size = int(max(to_x - from_x, to_y - from_y) * 1.5)

    new_from_x = max(from_x - 0.5 * margin, 0)
    new_from_y = max(from_y - 0.5 * margin, 0)
    new_to_x = min(new_from_x + size, 1024)
    new_to_y = min(new_from_y + size, 576)

    return [new_from_x, new_from_y, new_to_x - new_from_x, new_to_y - new_from_y]


while True:
    img0 = cam0.get_image()
    img0 = pygame.transform.flip(img0, True, False)
    img1 = cam1.get_image()
    img1 = pygame.transform.flip(img1, True, False)

    #    annotated_image_0, direction_score_0, face_locations_0 = face_inference(
    #        face_model_0, img0, timestamp
    #    )
    #    timestamp += 1
    face_locations = face_detection_inference(face_dedector_model, img0, timestamp)
    timestamp += 1

    #    annotated_image_1, direction_score_1, face_locations_1 = face_inference(
    #        face_model_1, img1, timestamp
    #    )
    #    timestamp += 1

    gameDisplay.blit(img0, (0, 0))
    if face_locations is not None and len(face_locations) > 0:
        for face in face_locations:
            face = convert_coord_draw_rect(face)
            pygame.draw.rect(gameDisplay, (255, 255, 255), face, 2)

    #    gameDisplay.blit(annotated_image_1, (1024, 0))
    #    gameDisplay.blit(
    #        font.render(f"DirectionScore: {direction_score_1}", True, white), (1024, 0)
    #    )

    # croped_0 = crop_resize_face(img0, face_locations_0)
    if face_locations is not None and len(face_locations) > 0:
        for idx, face in enumerate(face_locations):
            face = convert_coord_drop(face)
            croped_surface = img0.subsurface(face)
            scaled_surface = pygame.transform.scale(croped_surface, (576, 576))
            gameDisplay.blit(scaled_surface, (0 + idx * 576, 576))

            # gameDisplay.blit(img0, (0, 576), (300, 300, 100, 100))

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cam0.stop()
            cam1.stop()
            pygame.quit()
            exit()
