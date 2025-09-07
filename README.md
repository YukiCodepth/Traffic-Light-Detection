# Real-Time Traffic Light Detection

## Description
This project detects traffic light states (RED, YELLOW, GREEN) in real-time from a webcam or video using OpenCV and HSV color segmentation. It highlights detected lights with bounding boxes and labels.

## Features
- Detects RED, YELLOW, GREEN traffic lights
- Real-time detection with live visualization
- Auto-resizes frames for smoother processing
- Saves sample frames in the `samples/` folder
- Saves annotated video as `annotated.mp4` (optional)
- Live HSV trackbars to tune detection thresholds

## Requirements
- Python 3.8+
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

You can install all dependencies using:
```bash
pip install -r requirements.txt