import cv2
import time

import mediapipe as mp
from mediapipe.tasks.python import vision


def detector_callback(result: vision.FaceDetectorResult, image: mp.Image, n):
    for e in result.detections:
        print("bounding: ", e.bounding_box)


model_path = "./model/blaze_face_short_range.tflite"
options = vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=detector_callback,
)

with vision.FaceDetector.create_from_options(options) as detector:
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
        cv2.imshow("win", frame)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detector.detect_async(mp_image, int(time.time() * 1000))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()
