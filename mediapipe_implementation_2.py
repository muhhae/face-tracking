import cv2
import time

import mediapipe as mp
from mediapipe.tasks.python import vision


model_path = "./model/blaze_face_short_range.tflite"

base_options = mp.tasks.BaseOptions
face_detector = mp.tasks.vision.FaceDetector
face_detector_options = mp.tasks.vision.FaceDetectorOptions
vision_running_mode = mp.tasks.vision.RunningMode


options = face_detector_options(
    base_options=base_options(model_asset_path=model_path),
    running_mode=vision_running_mode.IMAGE,
)

with face_detector.create_from_options(options) as detector:
    vid = cv2.VideoCapture(0)

    start_time = time.time()
    frame_counter = 0

    while 1:
        frame_counter += 1

        ret, frame = vid.read()
        if ret is False:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fps = frame_counter / (time.time() - start_time)
        cv2.putText(
            frame,
            f"FPS: {fps:.3f}",
            (30, 30),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)

        for face in result.detections:
            b = face.bounding_box
            cv2.rectangle(
                frame,
                (b.origin_x, b.origin_y),
                (b.origin_x + b.width, b.origin_y + b.height),
                color=(255, 0, 0),
                thickness=2,
            )

        cv2.imshow("win", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()
