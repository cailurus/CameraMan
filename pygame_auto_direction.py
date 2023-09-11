import pygame
import pygame.camera
from cameraman.model import GestureModel, FaceModel, FaceDetectorModel, SoundModel
from cameraman.utils import (
    put_text,
    draw_face_landmarks,
    draw_hand,
    parse_audio_result,
    draw_face_direction,
)
import mediapipe as mp

# gesture_model = GestureModel()
face_model_0 = FaceModel()
face_model_1 = FaceModel()
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
    _, score = draw_face_direction(annotated_image, face_keypoints_result)
    annotated_image = pygame.surfarray.make_surface(annotated_image)

    return annotated_image, score


while True:
    img0 = cam0.get_image()
    img0 = pygame.transform.flip(img0, True, False)
    img1 = cam1.get_image()
    img1 = pygame.transform.flip(img1, True, False)

    annotated_image_0, direction_score_0 = face_inference(face_model_0, img0, timestamp)
    timestamp += 1
    annotated_image_1, direction_score_1 = face_inference(face_model_1, img1, timestamp)
    timestamp += 1

    gameDisplay.blit(annotated_image_0, (0, 0))
    gameDisplay.blit(
        font.render(f"DirectionScore: {direction_score_0}", True, white), (0, 0)
    )

    gameDisplay.blit(annotated_image_1, (1024, 0))
    gameDisplay.blit(
        font.render(f"DirectionScore: {direction_score_1}", True, white), (1024, 0)
    )

    if direction_score_0 < direction_score_1:
        gameDisplay.blit(annotated_image_0, (0, 576))
    else:
        gameDisplay.blit(annotated_image_1, (0, 576))

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cam0.stop()
            cam1.stop()
            pygame.quit()
            exit()
