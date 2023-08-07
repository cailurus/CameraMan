import cv2


def draw_gesture(image, gesture):
    cv2.putText(
        img=image,
        text=gesture,
        org=(0, 80),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.0,
        color=(255, 255, 255),
        thickness=3,
    )
    return image
