import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

VisionRunningMode = mp.tasks.vision.RunningMode


class GestureModel:
    def __init__(self):
        self.guesture = "No Hand Detected"
        self._setup()

    def parse_result(
        self,
        result: PoseLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        if result.gestures:
            self.guesture = result.gestures[0][0].category_name
        else:
            self.guesture = "No Hand Detected"

    def _setup(self):
        GestureOptions = GestureRecognizerOptions(
            base_options=BaseOptions(
                model_asset_path="./models/gesture_recognizer.task"
            ),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.parse_result,
        )
        self.model = GestureRecognizer.create_from_options(GestureOptions)

    def inference(self, image, timestamp):
        self.model.recognize_async(image, timestamp)
        return self.guesture
