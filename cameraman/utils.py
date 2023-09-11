import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from typing import Tuple, Union
import statistics
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

import numpy as np
import mediapipe as mp
import math


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels

FONT_SIZE = 1
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (255, 255, 255)  # vibrant green


def put_text(image, text, position):
    cv2.putText(
        img=image,
        text=f"{text}",
        org=position,
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=FONT_SIZE,
        color=(255, 255, 255),
        thickness=FONT_THICKNESS,
    )
    return image


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def get_boundingbox_detector(image, detection_result) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    if detection_result.detections is None:
        return None
    frame_height, frame_width, c = image.shape
    face_locations = []

    for detection in detection_result.detections:
        from_x_relative = detection.location_data.relative_bounding_box.xmin
        from_y_relative = detection.location_data.relative_bounding_box.ymin
        width_relative = detection.location_data.relative_bounding_box.width
        height_relative = detection.location_data.relative_bounding_box.height
        # Draw bounding_box
        # face_react = np.multiply(
        #     [
        #         detection.location_data.relative_bounding_box.xmin,
        #         detection.location_data.relative_bounding_box.ymin,
        #         detection.location_data.relative_bounding_box.width,
        #         detection.location_data.relative_bounding_box.height,
        #     ],
        #     [frame_width, frame_height, frame_width, frame_height],
        # ).astype(int)
        from_x = int(from_x_relative * frame_width)
        from_y = int(from_y_relative * frame_height)
        to_x = int(
            from_x + width_relative * frame_width,
        )
        to_y = int(from_y + height_relative * frame_height)
        # face_react = [
        #     detection.location_data.relative_bounding_box.xmin * frame_width,
        #     detection.location_data.relative_bounding_box.ymin * frame_height,
        #     detection.location_data.relative_bounding_box.xmin * frame_width
        #     + detection.location_data.relative_bounding_box.width * frame_width,
        #     detection.location_data.relative_bounding_box.ymin * frame_height
        #     + detection.location_data.relative_bounding_box.height * frame_height,
        # ]

        face_locations.append([from_x, from_y, to_x, to_y])
        # image = cv2.rectangle(image, (from_x, from_y), (to_x, to_y), (255, 255, 255), 2)

        # cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # # Draw keypoints
        # for keypoint in detection.keypoints:
        #     # keypoint_px = _normalized_to_pixel_coordinates(
        #     #    keypoint.x, keypoint.y, width, height
        #     # )
        #     keypoint_px = _normalized_to_pixel_coordinates(
        #         keypoint.x, keypoint.y, width, height
        #     )

        #     color, thickness, radius = (0, 255, 0), 2, 2
        #     cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # # Draw label and score
        # category = detection.categories[0]
        # category_name = category.category_name
        # category_name = "" if category_name is None else category_name
        # probability = round(category.score, 2)
        # result_text = category_name + " (" + str(probability) + ")"
        # text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        # cv2.putText(
        #     annotated_image,
        #     result_text,
        #     text_location,
        #     cv2.FONT_HERSHEY_PLAIN,
        #     FONT_SIZE,
        #     TEXT_COLOR,
        #     FONT_THICKNESS,
        # )

    return face_locations


def draw_face_direction(rgb_image, detection_result):
    if detection_result is None:
        return rgb_image, 0

    face_landmarks_list = detection_result.face_landmarks
    img_h, img_w, img_c = rgb_image.shape
    annotated_image = np.copy(rgb_image)

    direction_scores = []
    face_2d, face_3d = [], []
    for face_idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[face_idx]
        for idx, lm in enumerate(face_landmarks):
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

        #         if y < -threshold:
        #             text = "Looking Left"
        #         elif y > threshold:
        #             text = "Looking Right"
        #         elif x < -threshold:
        #             text = "Looking Down"
        #         elif x > threshold:
        #             text = "Looking Up"
        #         else:
        #             text = "Forward"

        nose_3d_projection, jacobian = cv2.projectPoints(
            nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix
        )

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
        direction_score = math.sqrt(x**2 + y**2)

        #        annotated_image = put_text(annotated_image, text, (50, 100))

        #        annotated_image = put_text(
        #            annotated_image, f"Direction Score: {direction_score:.2f}", (50, 130)
        #        )

        cv2.arrowedLine(annotated_image, p1, p2, (255, 255, 255), 3, tipLength=0.2)
        direction_scores.append(direction_score)
    if len(direction_scores) == 0:
        avg_direction_score = 100
    else:
        avg_direction_score = statistics.mean(direction_scores)
    return (annotated_image, avg_direction_score)


def draw_face_landmarks(rgb_image, detection_result):
    if detection_result is None:
        return rgb_image, [[0, 0, 0, 0]]

    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    face_locations = []
    print(len(face_landmarks_list))
    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

        x_coordinates = [landmark.x for landmark in face_landmarks]
        y_coordinates = [landmark.y for landmark in face_landmarks]
        from_x = int(min(x_coordinates) * width)
        from_y = int(min(y_coordinates) * height)
        to_x = int(max(x_coordinates) * width)
        to_y = int(max(y_coordinates) * height)
        face_locations.append((from_x, from_y, to_x, to_y))
    #        cv2.rectangle(
    #            annotated_image,
    #            (from_x - MARGIN, from_y - MARGIN),
    #            (to_x + MARGIN, to_y + MARGIN),
    #            (0, 255, 0),
    #            2,
    #        )

    return annotated_image, face_locations


def draw_hand(rgb_image, detection_result, gesture_only=True):
    if detection_result is None:
        return rgb_image
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    gestures_list = detection_result.gestures

    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        gesture = gestures_list[idx]

        if gesture_only is False:
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style(),
            )

        # Get the top left corner of the detected hand's bounding box.
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name} {gesture[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


def draw_face_box(image, detection_result) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    if detection_result is None:
        return image

    annotated_image = image.copy()
    height, width, _ = image.shape
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, (255, 0, 0), 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(
                keypoint.x, keypoint.y, width, height
            )
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

            # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = "" if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + " (" + str(probability) + ")"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(
            annotated_image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            (255, 0, 0),
            FONT_THICKNESS,
        )

    return annotated_image


def parse_audio_result(result):
    if result:
        classification = result.classifications[0]
        label_list = [category.category_name for category in classification.categories]
        print(label_list)
        result_str = ", ".join(label_list)

        return result_str
    else:
        return ""
