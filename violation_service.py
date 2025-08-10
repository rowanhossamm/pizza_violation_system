import pika
import json
import sqlite3
from utils import is_inside_partial, boxes_iou


'''
roi_boxes = [
    (475, 265, 525, 305),
    (460, 305, 510, 345),   
    (455, 345, 505, 385),   
    (445, 395, 495, 435),  
    (435, 440, 485, 480),   
    (425, 480, 475, 525),   
    (415, 530, 465, 580),   
    (405, 580, 455, 630), 
    (395, 635, 445, 685),  
]
'''


# ROI boxes 
roi_boxes = [
    (400, 265, 450, 300),
    (515, 265, 565, 300),
    (390, 305, 440, 335),   
    (375, 340, 430, 375),   
    (355, 385, 410, 430),  
    (345, 440, 395, 475),   
    (335, 480, 385, 520),   
    (325, 525, 380, 575),   
    (315, 580, 365, 625), 
    (305, 635, 355, 680),  
]

violation_count = 0
hand_states = {}

CLASS_NAMES = {0: "hand", 1: "person", 2: "pizza", 3: "scooper"}

# Setup SQLite connection and create table if it doesn't exist
conn = sqlite3.connect('violations.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    hand_bbox TEXT NOT NULL,
    roi_bbox TEXT NOT NULL,
    violation_type TEXT,
    notes TEXT
)
''')
conn.commit()

def save_violation_to_db(frame_id, hand_box, roi_box, violation_type="No scooper used", notes=""):
    hand_str = ",".join(map(str, map(int, hand_box)))
    roi_str = ",".join(map(str, map(int, roi_box)))

    cursor.execute('''
        INSERT INTO violations (frame_id, hand_bbox, roi_bbox, violation_type, notes)
        VALUES (?, ?, ?, ?, ?)
    ''', (frame_id, hand_str, roi_str, violation_type, notes))
    conn.commit()

def round_position(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2) // 20
    cy = int((y1 + y2) / 2) // 20
    return (cx, cy)

def is_near(box1, box2, threshold=0.4):
    return boxes_iou(box1, box2) > threshold

def detect_violation(detections, frame_id):
    global violation_count
    new_violations = []
    hands = []
    scoopers = []

    # Separate hands and scoopers from detections
    for det in detections:
        cls = det["class"]
        box = det["bbox"]
        track_id = det.get("id", round_position(box))

        if cls == 0:  # hand
            hands.append((box, track_id))
        elif cls == 3:  # scooper
            scoopers.append(box)

    # Check each hand for violation inside ROIs
    for hand_box, key_id in hands:
        if key_id not in hand_states:
            hand_states[key_id] = {"inside": False}

        inside_any_roi = False
        current_roi = None
        for roi in roi_boxes:
            if is_inside_partial(hand_box, roi):
                inside_any_roi = True
                current_roi = roi
                break

        holding_scooper = any(is_near(hand_box, scooper) for scooper in scoopers)

        print(f"[DEBUG] Frame {frame_id} - Hand {key_id} inside ROI: {inside_any_roi}, holding scooper: {holding_scooper}")

        if inside_any_roi and not holding_scooper:
            if not hand_states[key_id]["inside"]:
                violation_count += 1
                hand_states[key_id]["inside"] = True
                new_violation = {
                    "frame_id": frame_id,
                    "hand": hand_box,
                    "roi": current_roi
                }
                new_violations.append(new_violation)
                # Save violation to DB
                save_violation_to_db(frame_id, hand_box, current_roi)
        else:
            if not inside_any_roi:
                hand_states[key_id]["inside"] = False

    return new_violations

# RabbitMQ connection setup
connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue="detection")

def callback(ch, method, properties, body):
    message = json.loads(body)
    frame_id = message["frame_id"]
    detections = message["detections"]

    # Debug print detections
    print(f"\n[STREAMING DATA] Frame {frame_id}:")
    for d in detections:
        cls_name = CLASS_NAMES.get(d["class"], f"cls_{d['class']}")
        print(f" - ID: {d.get('id', 'N/A')} | Class: {cls_name} | BBox: {d['bbox']}")

    violations = detect_violation(detections, frame_id)
    if violations:
        print(f"[Frame {frame_id}] New Violations Detected:")
        for v in violations:
            print(f" - Hand: {v['hand']} in ROI {v['roi']}")

channel.basic_consume(queue="detection", on_message_callback=callback, auto_ack=True)

print("Violation Service is running and listening to detection queue...")

import atexit
atexit.register(lambda: conn.close())  # Ensure DB connection closes on exit

channel.start_consuming()
