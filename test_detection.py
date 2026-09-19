import cv2

from modules.camera import open_camera, read_frame, release_camera
from modules.detection import ObjectDetector
from modules.alerts import send_voice_alerts


camera = open_camera()
detector = ObjectDetector("yolo11n.pt")

print("AI Vision Assistant started.")
print("Press Q to quit.")

while True:

    frame = read_frame(camera)

    if frame is None:
        print("Camera frame nahi mil raha.")
        break

    detections = detector.detect(frame)

    # Bounding boxes
    for detection in detections:

        object_name = detection["object"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = detection["box"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"{object_name} {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Voice alerts
    if detections:
        send_voice_alerts(
            detections,
            frame.shape[1],
            frame.shape[0]
        )

    cv2.imshow("AI Vision Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


release_camera(camera)
print("AI Vision Assistant stopped.")