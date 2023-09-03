import pygame
import pygame.camera
from cameraman.model import GestureModel, FaceModel, FaceDetectorModel, SoundModel
from cameraman.utils import put_text, draw_face_landmarks, draw_hand, parse_audio_result
import mediapipe as mp


# gesture_model = GestureModel()
face_model = FaceModel()
# face_detector_model = FaceDetectorModel()

pygame.init()

gameDisplay = pygame.display.set_mode((1920, 1080), 0)

pygame.camera.init()
print(pygame.camera.list_cameras())

cam = pygame.camera.Camera(0, (500, 500))
# cam = pygame.camera.Camera(self.clist[0], self.size)

cam.start()
timestamp = 0

while True:
    img = cam.get_image()
    img = pygame.transform.flip(img, True, False)
    gameDisplay.blit(img, (0, 0))

    imgdata = pygame.surfarray.array3d(img)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgdata)

    face_result = face_model.inference(mp_image, timestamp)

    annotated_image = draw_face_landmarks(imgdata, face_result)

    surf = pygame.surfarray.make_surface(annotated_image)

    gameDisplay.blit(surf, (500, 500))

    pygame.display.update()
    timestamp += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cam.stop()
            pygame.quit()
            exit()
