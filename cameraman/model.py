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


class DetectionBase:
    def __init__(self):
        self.result = None

    def parse_result(self, result, output_image, timestamp_ms):
        self.result = result

    def inference(self, image, timestamp):
        raise NotImplementedError


class GestureModel(DetectionBase):
    def __init__(self):
        super(GestureModel, self).__init__()
        self.setup()

    def setup(self):
        GestureOptions = GestureRecognizerOptions(
            base_options=BaseOptions(
                model_asset_path="./models/gesture_recognizer.task"
            ),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=self.parse_result,
        )
        self.model = GestureRecognizer.create_from_options(GestureOptions)

    def inference(self, image, timestamp):
        self.model.recognize_async(image, timestamp)
        return self.result


class FaceModel(DetectionBase):
    def __init__(self):
        self.result = None
        self.setup()

    def setup(self):
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
        return self.result


class FaceDetectorModel(DetectionBase):
    def __init__(self):
        self.result = None
        self.setup()

    def setup(self):
        base_options = FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path="./models/blaze_face_short_range.tflite"
            ),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.parse_result,
        )

        self.model = FaceDetector.create_from_options(base_options)

    def inference(self, image, timestamp):
        self.model.detect_async(image, timestamp)
        return self.result
