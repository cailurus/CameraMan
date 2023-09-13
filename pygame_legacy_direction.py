import pygame
import pygame.camera
import mediapipe as mp
import statistics
import cv2
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
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


def face_inference(face_mesh_model, img):
    direction_scores = []
    avg_direction_score = 0
    face_2d, face_3d = [], []
    view = pygame.surfarray.array3d(img)
    view = view.transpose([1, 0, 2])
    # image = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
    image = view.copy()
    img_h, img_w, img_c = image.shape
    print(img_h, img_w, img_c)
    results = face_mesh_model.process(image)

    image.flags.writeable = True
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

            for idx, lm in enumerate(face_landmarks.landmark):
                if (
                    idx == 33
                    or idx == 263
                    or idx == 1
                    or idx == 61
                    or idx == 291
                    or idx == 199
                ):
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 5000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * img_w
            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d,
                face_2d,
                cam_matrix,
                dist_matrix,
            )
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            threshold = 6

            if y < -threshold:
                text = "Looking Left"
            elif y > threshold:
                text = "Looking Right"
            elif x < -threshold:
                text = "Looking Down"
            elif x > threshold:
                text = "Looking Up"
            else:
                text = "Forward"

            nose_3d_projection, jacobian = cv2.projectPoints(
                nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix
            )
            print(text)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            direction_score = math.sqrt(x**2 + y**2)
            cv2.arrowedLine(image, p1, p2, (255, 255, 255), 3, tipLength=0.2)
            direction_scores.append(direction_score)

    if len(direction_scores) == 0:
        avg_direction_score = 100
    else:
        avg_direction_score = statistics.mean(direction_scores)

    image = image.transpose([1, 0, 2])
    return image, avg_direction_score


while True:
    img0 = cam0.get_image()
    img0 = pygame.transform.flip(img0, True, False)
    img1 = cam1.get_image()
    img1 = pygame.transform.flip(img1, True, False)

    annotated_image_0, direction_score_0 = face_inference(model0, img0)
    annotated_image_0 = pygame.surfarray.make_surface(annotated_image_0)
    gameDisplay.blit(annotated_image_0, (0, 0))

    gameDisplay.blit(
        font.render(f"DirectionScore: {direction_score_0}", True, white), (0, 0)
    )

    annotated_image_1, direction_score_1 = face_inference(model1, img1)
    annotated_image_1 = pygame.surfarray.make_surface(annotated_image_1)
    gameDisplay.blit(annotated_image_1, (1024, 0))

    gameDisplay.blit(
        font.render(f"DirectionScore: {direction_score_1}", True, white), (1024, 0)
    )

    if direction_score_0 < direction_score_1:
        gameDisplay.blit(annotated_image_0, (0, 576))
    else:
        gameDisplay.blit(annotated_image_1, (0, 576))
    #
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cam0.stop()
            cam1.stop()
            pygame.quit()
            exit()
