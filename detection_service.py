from ultralytics import YOLO
import cv2
import pika
import json

video_path = "videos/wrong.mp4"
model = YOLO("best.pt")

connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue="detection")

cap = cv2.VideoCapture(video_path)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = box
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class": int(cls),
            "score": score
        })

    message = {
        "frame_id": frame_id,
        "detections": detections
    }

    channel.basic_publish(
        exchange="",
        routing_key="detection",
        body=json.dumps(message)
    )

    frame_id += 1

cap.release()
connection.close()