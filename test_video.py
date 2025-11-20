# --- test_video.py ---
import cv2
import time
from recievers.webCamSim import VideoThreadedCapture

# --- Configuration ---
video_path = 0  # Use 0 for live webcam, or path to video file
loop_video = True  # Only relevant if using a video file

# --- Initialize capture ---
cap = VideoThreadedCapture(0)

if not cap.isOpened():
    print("Error: Could not open video/camera.")
    exit()

cv2.namedWindow("Test Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Test Video", 640, 480)

prev_time = time.time()
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame not available yet...")
            time.sleep(0.01)
            continue

        # Optional: show FPS
        frame_count += 1
        current_time = time.time()
        if current_time - prev_time >= 1.0:
            print(f"Loops per second: {frame_count}")
            frame_count = 0
            prev_time = current_time

        cv2.imshow("Test Video", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
