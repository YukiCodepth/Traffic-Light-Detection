import cv2
import numpy as np
import os

# ---------------- CONFIGURATION ----------------
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
RESIZE_WIDTH, RESIZE_HEIGHT = 640, 480
SAVE_SAMPLES_DIR = 'samples'
SAVE_VIDEO = True
ANNOTATED_VIDEO = 'annotated.mp4'
MIN_AREA = 100

# Create folder for sample frames
if not os.path.exists(SAVE_SAMPLES_DIR):
    os.makedirs(SAVE_SAMPLES_DIR)

# ---------------- HELPER FUNCTIONS ----------------
def nothing(x):
    pass

def detect_color(mask, frame, label, color_bgr):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

# ---------------- VIDEO SETUP ----------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Cannot open video source: {VIDEO_SOURCE}")
    exit()

# Video writer
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ANNOTATED_VIDEO, fourcc, 20.0, (RESIZE_WIDTH, RESIZE_HEIGHT))

# ---------------- TRACKBARS FOR HSV ----------------
cv2.namedWindow("HSV Trackbars")
cv2.resizeWindow("HSV Trackbars", 400, 300)

# Red range
cv2.createTrackbar("LowH_R1","HSV Trackbars",0,180,nothing)
cv2.createTrackbar("HighH_R1","HSV Trackbars",10,180,nothing)
cv2.createTrackbar("LowH_R2","HSV Trackbars",160,180,nothing)
cv2.createTrackbar("HighH_R2","HSV Trackbars",180,180,nothing)
cv2.createTrackbar("LowS_R","HSV Trackbars",150,255,nothing)
cv2.createTrackbar("HighS_R","HSV Trackbars",255,255,nothing)
cv2.createTrackbar("LowV_R","HSV Trackbars",50,255,nothing)
cv2.createTrackbar("HighV_R","HSV Trackbars",255,255,nothing)

# Yellow range
cv2.createTrackbar("LowH_Y","HSV Trackbars",15,180,nothing)
cv2.createTrackbar("HighH_Y","HSV Trackbars",40,180,nothing)
cv2.createTrackbar("LowS_Y","HSV Trackbars",120,255,nothing)
cv2.createTrackbar("HighS_Y","HSV Trackbars",255,255,nothing)
cv2.createTrackbar("LowV_Y","HSV Trackbars",120,255,nothing)
cv2.createTrackbar("HighV_Y","HSV Trackbars",255,255,nothing)

# Green range
cv2.createTrackbar("LowH_G","HSV Trackbars",40,180,nothing)
cv2.createTrackbar("HighH_G","HSV Trackbars",90,180,nothing)
cv2.createTrackbar("LowS_G","HSV Trackbars",50,255,nothing)
cv2.createTrackbar("HighS_G","HSV Trackbars",255,255,nothing)
cv2.createTrackbar("LowV_G","HSV Trackbars",50,255,nothing)
cv2.createTrackbar("HighV_G","HSV Trackbars",255,255,nothing)

frame_count = 0

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame")
        break

    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV values from trackbars
    lowH_R1 = cv2.getTrackbarPos("LowH_R1","HSV Trackbars")
    highH_R1 = cv2.getTrackbarPos("HighH_R1","HSV Trackbars")
    lowH_R2 = cv2.getTrackbarPos("LowH_R2","HSV Trackbars")
    highH_R2 = cv2.getTrackbarPos("HighH_R2","HSV Trackbars")
    lowS_R = cv2.getTrackbarPos("LowS_R","HSV Trackbars")
    highS_R = cv2.getTrackbarPos("HighS_R","HSV Trackbars")
    lowV_R = cv2.getTrackbarPos("LowV_R","HSV Trackbars")
    highV_R = cv2.getTrackbarPos("HighV_R","HSV Trackbars")

    lowH_Y = cv2.getTrackbarPos("LowH_Y","HSV Trackbars")
    highH_Y = cv2.getTrackbarPos("HighH_Y","HSV Trackbars")
    lowS_Y = cv2.getTrackbarPos("LowS_Y","HSV Trackbars")
    highS_Y = cv2.getTrackbarPos("HighS_Y","HSV Trackbars")
    lowV_Y = cv2.getTrackbarPos("LowV_Y","HSV Trackbars")
    highV_Y = cv2.getTrackbarPos("HighV_Y","HSV Trackbars")

    lowH_G = cv2.getTrackbarPos("LowH_G","HSV Trackbars")
    highH_G = cv2.getTrackbarPos("HighH_G","HSV Trackbars")
    lowS_G = cv2.getTrackbarPos("LowS_G","HSV Trackbars")
    highS_G = cv2.getTrackbarPos("HighS_G","HSV Trackbars")
    lowV_G = cv2.getTrackbarPos("LowV_G","HSV Trackbars")
    highV_G = cv2.getTrackbarPos("HighV_G","HSV Trackbars")

    # Masks
    red_mask = cv2.inRange(hsv, np.array([lowH_R1, lowS_R, lowV_R]), np.array([highH_R1, highS_R, highV_R])) | \
               cv2.inRange(hsv, np.array([lowH_R2, lowS_R, lowV_R]), np.array([highH_R2, highS_R, highV_R]))
    yellow_mask = cv2.inRange(hsv, np.array([lowH_Y, lowS_Y, lowV_Y]), np.array([highH_Y, highS_Y, highV_Y]))
    green_mask = cv2.inRange(hsv, np.array([lowH_G, lowS_G, lowV_G]), np.array([highH_G, highS_G, highV_G]))

    # Morphology to remove noise
    kernel = np.ones((3,3), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_DILATE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_DILATE, kernel)

    # Detect colors
    detect_color(red_mask, frame, 'RED', (0,0,255))
    detect_color(yellow_mask, frame, 'YELLOW', (0,255,255))
    detect_color(green_mask, frame, 'GREEN', (0,255,0))

    # Show frame
    cv2.imshow("Traffic Light Detection", frame)

    # Save sample frames
    if frame_count % 30 == 0:
        cv2.imwrite(os.path.join(SAVE_SAMPLES_DIR, f"frame_{frame_count}.jpg"), frame)

    # Save video
    if SAVE_VIDEO:
        out.write(frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if SAVE_VIDEO:
    out.release()
cv2.destroyAllWindows()