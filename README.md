# Pizza Store Scooper Violation Detection System

## Overview
This project detects hygiene violations in a pizza store by monitoring if workers pick ingredients without using a scooper from designated Regions of Interest (ROIs). Violations are flagged when a hand enters an ROI and picks an ingredient without a scooper.


## Implemented Components

### 1. Detection Service (`detection.py`)
- Reads video frames from a video file.
- Runs object detection using a pretrained YOLO model to detect hands, scoopers, pizza, and people.
- Publishes detection results (bounding boxes, classes, confidence) to RabbitMQ queue `detection`.

### 2. Violation Detection Service (`violation.py`)
- Subscribes to detection results from RabbitMQ.
- Applies violation rules based on whether a hand is inside an ROI and holding a scooper.
- Counts and logs violations.
- Saves violation records into a SQLite database (`violations.db`) with details (frame ID, bounding boxes, ROI, timestamp).

### 3. Utilities (`utils.py`)
- Helper functions for bounding box overlap (IOU) calculations and partial ROI checks.

### 4. Tracking (`sort.py`)
- SORT tracking algorithm to track bounding boxes across frames.
- Not fully integrated yet but available to improve hand tracking.

### 5. Frontend UI (`index.html`)
- Simple webpage displaying the video stream with bounding boxes and violation highlights.

---

## How to Run

1. Make sure RabbitMQ server is running on `localhost`.
2. Run utlis.py and sort.py
3. Run the Detection_Service.py
4. Run the Violation Detection Service in another terminal:
   python violation.py
6. Run streaming_service.py in another terminal:
   python sreaming_service.py
7. Open index.html in a web browser (assuming streaming service implemented).
8. Watch console logs for detections and violations.

## Dependencies
Python 3.8+
RabbitMQ server on localhost
Python libraries (install via requirements.txt)

## Installation
Install Python dependencies:
pip install -r requirements.txt

## Notes
Violations are saved locally in violations.db.
Detailed logs printed in console.
Tracking via SORT is ready but not fully integrated yet.
Dockerization and dedicated frame reader service are future improvements.


## Demo Video
Watch the demo video here: (https://drive.google.com/file/d/1MZpkFFc-p88p5YcnwHo0SLVCRo50eNO0/view?usp=sharing))
