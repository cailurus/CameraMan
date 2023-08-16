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


AudioClassifier = mp.tasks.audio.AudioClassifier
AudioClassifierOptions = mp.tasks.audio.AudioClassifierOptions
AudioClassifierResult = mp.tasks.audio.AudioClassifierResult
AudioRunningMode = mp.tasks.audio.RunningMode

from mediapipe.tasks.python.components import containers

buffer_size, sample_rate, num_channels = 15600, 16000, 1
audio_format = containers.AudioDataFormat(num_channels, sample_rate)
audio_data = containers.AudioData(buffer_size, audio_format)


class SoundModel(DetectionBase):
    def __init__(self):
        super(SoundModel, self).__init__()
        self.setup()

    def parser_result(self, result: AudioClassifierResult, timestamp_ms: int):
        self.result = result

    def setup(self):
        BaseOptions = mp.tasks.BaseOptions
        options = AudioClassifierOptions(
            base_options=BaseOptions(model_asset_path="./models/yamnet.tflite"),
            running_mode=AudioRunningMode.AUDIO_STREAM,
            max_results=3,
            score_threshold=0.3,
            result_callback=self.parser_result,
        )
        self.model = AudioClassifier.create_from_options(options)

    def inference(self, audio_raw, timestamp):
        # Load the input audio from the AudioRecord instance and run classify.
        # audio_data.load_from_array(data.astype(np.float32))

        audio_data.load_from_array(audio_raw)
        self.model.classify_async(audio_data, timestamp)
        return self.result


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
