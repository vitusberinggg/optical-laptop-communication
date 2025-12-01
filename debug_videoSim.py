import cv2
import time
import numpy as np
from utilities.color_functions_v3_1 import build_color_LUT
from utilities.bitgrid_debugger import debug_bitgrid_realtime
from webcam_simulation.webcamSimulator import VideoThreadedCapture

corrected_ranges = {
    "red":    (np.array([0, 100, 100]), np.array([10, 255, 255])),  # hue 0–10
    "red2":   (np.array([160, 100, 100]), np.array([179, 255, 255])),  # hue 160–179
    "white":  (np.array([0, 0, 220]), np.array([180, 25, 255])),
    "black":  (np.array([0, 0, 0]), np.array([180, 255, 35])),
    "green":  (np.array([45, 80, 80]), np.array([75, 255, 255])),
    "blue":   (np.array([95, 120, 70]), np.array([130, 255, 255]))
}
LUT, color_names = build_color_LUT(corrected_ranges)

cap = VideoThreadedCapture(r"C:\Users\ejadmax\code\optical-laptop-communication\webcam_simulation\sender_v3_video.mp4")

if not cap.isOpened():

        print("Error: Could not open camera/video.")
        exit()

while True:

        ret, frame = cap.read()

        if ret:
            break
        time.sleep(0.01)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    debug_bitgrid_realtime(hsv, LUT=LUT, color_names=color_names)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
