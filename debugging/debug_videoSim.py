import cv2
from utilities.color_functions_v3_1 import build_color_LUT
from utilities.bitgrid_debugger import debug_bitgrid_realtime
from webcam_simulation.webcamSimulator import VideoThreadedCapture

tracker_colors = {
    'black': ([0,0,0],[180,255,50]),
    'white': ([0,0,200],[180,30,255])
}
LUT, color_names = build_color_LUT(tracker_colors)

cap = cv2.VideoCapture(r"C:\Users\ejadmax\code\optical-laptop-communication\webcam_simulation\sender_v3_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    debug_bitgrid_realtime(hsv, LUT=LUT, color_names=color_names)

    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
