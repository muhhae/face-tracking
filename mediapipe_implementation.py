import cv2
import time

import mediapipe as mp
from mediapipe.tasks.python import vision

last_timestamp = 0
current_bounding_boxes = []


def detector_callback(
    result: vision.FaceDetectorResult, image: mp.Image, timestamp: int
):
    global last_timestamp, current_bounding_boxes
    if last_timestamp > timestamp:
        print("Last timestamp is greater than current timestamp")
        return
    last_timestamp = timestamp
    current_bounding_boxes = [f.bounding_box for f in result.detections]


model_path = "./model/blaze_face_short_range.tflite"
options = vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=detector_callback,
)

with vision.FaceDetector.create_from_options(options) as detector:
    start_time = time.time()
    frame_counter = 0
    vid = cv2.VideoCapture(0)

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
        detector.detect_async(mp_image, int(time.time() * 1000))

        for b in current_bounding_boxes:
            cv2.rectangle(
                frame,
                (b.origin_x, b.origin_y),
                (b.origin_x + b.width, b.origin_y + b.height),
                color=(0, 255, 255),
                thickness=2,
            )

        cv2.imshow("win", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()
