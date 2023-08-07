import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

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


class FaceModel:
    def __init__(self):
        self.face_landmarks = None
        self._setup()

    def parse_result(
        self, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ):
        if result.face_landmarks:
            self.face_landmarks = result.face_landmarks
        else:
            self.face_landmarks = None

    def _setup(self):
        base_options = BaseOptions(model_asset_path="./models/face_landmarker.task")
        options = FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_faces=1,
            result_callback=self.parse_result,
        )
        self.model = FaceLandmarker.create_from_options(options)

    def inference(self, image, timestamp):
        self.model.detect_async(image, timestamp)
        return self.face_landmarks
