# streaming_service.py
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO
from utils import is_inside_partial, boxes_iou
from sort import Sort 

app = Flask(__name__)
model = YOLO("best.pt")


roi_boxes = [
    (400, 265, 450, 300),

    (390, 305, 440, 335), 
    (515, 265, 565, 300),  
    (375, 340, 430, 375),   
    (355, 385, 410, 430),  
    (345, 440, 395, 475),   
    (335, 480, 385, 520),   
    (325, 525, 380, 575),   
    (315, 580, 365, 625), 
    (305, 635, 355, 680),  
]

violation_count = 0

# {"inside": True/False}
hand_states = {}

tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.2)

video_path = "videos/wrong.mp4"
cap = cv2.VideoCapture(video_path)

def gen_frames():
    global violation_count, hand_states
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_index += 1
        results = model(frame)[0]

        detections = []
        scoopers = []

        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if label == "hand":
                detections.append([x1, y1, x2, y2, conf])
                color = (0, 255, 0)
            elif label == "scooper":
                scoopers.append((x1, y1, x2, y2))
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        
        if len(detections) > 0:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 5))

        
        tracked_hands = tracker.update(detections)

        
        for roi in roi_boxes:
            cv2.rectangle(frame, roi[:2], roi[2:], (0, 255, 255), 2)
            cv2.putText(frame, "ROI", (roi[0], roi[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        
        current_ids = set()
        for x1, y1, x2, y2, track_id in tracked_hands:
            track_id = int(track_id)
            hand_box = (int(x1), int(y1), int(x2), int(y2))
            inside_roi = any(is_inside_partial(hand_box, roi, 0.7) for roi in roi_boxes)
            holding_scooper = any(boxes_iou(hand_box, scooper) > 0.3 for scooper in scoopers)

            if track_id not in hand_states:
                hand_states[track_id] = {"inside": False}

            if inside_roi and not holding_scooper:
                if not hand_states[track_id]["inside"]:
                    violation_count += 1
                    hand_states[track_id]["inside"] = True
                    cv2.putText(frame, "VIOLATION!", (hand_box[0], hand_box[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    cv2.rectangle(frame, (hand_box[0], hand_box[1]),
                                  (hand_box[2], hand_box[3]), (0, 0, 255), 3)
            else:
                if not inside_roi:
                    hand_states[track_id]["inside"] = False

            current_ids.add(track_id)


        hand_states = {tid: state for tid, state in hand_states.items() if tid in current_ids}

        
        cv2.putText(frame, f"Violations: {violation_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/violations_count')
def violations_count():
    return jsonify({"violations": violation_count})

if __name__ == '__main__':
    app.run(debug=True)
