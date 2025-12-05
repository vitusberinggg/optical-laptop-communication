import cv2
import time
import numpy as np
import queue
from utilities.decoding_functions_v3_1 import decode_bitgrid
from utilities.color_functions_v3_1 import build_color_LUT, tracker
from bitgrid_debugger import debug_bitgrid_realtime
from webcam_simulation.webcamSimulator import VideoThreadedCapture
from reciever_v3_1 import decoding_worker

# function that pre-compiles the numba functions to prevent lag on initial launch with them
def warmup_all():
    from utilities.color_functions_v3_1 import bitgrid_majority_calculator

    # Warm up bitgrid majority calc
    dummy_merged = np.zeros((2, 2, 8, 16, 10), dtype=np.uint8)
    bitgrid_majority_calculator(dummy_merged, 5)

corrected_ranges = {
    "red":    (np.array([0, 100, 100]), np.array([10, 255, 255])),  # hue 0–10
    "red2":   (np.array([160, 100, 100]), np.array([179, 255, 255])),  # hue 160–179
    "white":  (np.array([0, 0, 220]), np.array([180, 25, 255])),
    "black":  (np.array([0, 0, 0]), np.array([180, 255, 35])),
    "green":  (np.array([45, 80, 80]), np.array([75, 255, 255])),
    "blue":   (np.array([95, 120, 70]), np.array([130, 255, 255]))
}
LUT, color_names = build_color_LUT(corrected_ranges)
tracker.colors(LUT, color_names)

warmup_all()


cap = VideoThreadedCapture(r"C:\Users\ejadmax\code\optical-laptop-communication\webcam_simulation\sender_v3_video.mp4")

if not cap.isOpened():

        print("Error: Could not open camera/video.")
        exit()

while True:

        ret, frame = cap.read()

        if ret:
            break
        time.sleep(0.01)


cv2.namedWindow("DEBUG", cv2.WINDOW_NORMAL)
cv2.resizeWindow("DEBUG", 1920, 1200)
cv2.namedWindow("OG", cv2.WINDOW_NORMAL)
cv2.resizeWindow("OG", 1920, 1200)
cv2.namedWindow("All Cells", cv2.WINDOW_NORMAL)
cv2.resizeWindow("All Cells", 1920, 1200)


frame = cv2.imread(r"C:\Users\ejadmax\code\optical-laptop-communication\debugging\debug_img.png")


#while True:
'''
ret, frame = cap.read()
if not ret:
    break
'''
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#overlay = debug_bitgrid_realtime(hsv, LUT=LUT, color_names=color_names)

track = time.time()
for i in range(6):
    from reciever_v3_1 import frame_queue

    add_frame = False
    end_frame = False
    recall = False

    if i < 4:
        add_frame = True
    elif i == 4:
        add_frame = True
        end_frame = True
    else:
        recall = True

    try:
        frame_queue.put_nowait((hsv.copy(), recall, add_frame, end_frame))
    except queue.Full:
        pass  # skip if queue is full

from reciever_v3_1 import decoded_message
message = decoded_message
print(f"[DEBUG] Time taken to process bitgrid: {time.time() - track:.3f} seconds")

print(f"Decoded message: {message}")

key = cv2.waitKey(1) & 0xFF
#if key == ord('q'):
    #break

cap.release()
cv2.destroyAllWindows()
