import cv2
import numpy as np
import sys
import os
from utilities.detection_functions import detect_aruco_marker_frame

VIDEO_PATH = r"C:\Users\eanpaln\Videos\Screen Recordings\Screen Recording 2025-11-14 150546.mp4"
SAVE_DETECTED_FRAME_AS = "detected_frame.png"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Could not open video:", VIDEO_PATH)
    sys.exit(1)

print("Video opened. Press 'q' to quit.")

found_and_saved = False
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    frame_idx += 1

    try:
        H = detect_aruco_marker_frame(frame)
    except Exception as e:
        print(f"Frame {frame_idx}: detect_aruco_marker_frame raised:", e)
        continue

    display = frame.copy()

    if H is None:
        cv2.putText(display, f"Frame {frame_idx}: No markers detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    else:
        print(f"\nFrame {frame_idx}: Homography matrix:\n{H}\n")
        cv2.putText(display, f"Frame {frame_idx}: Markers detected!",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Try to draw mapped rectangle
        try:
            dst = np.array([
                [0, 0, 1],
                [df_module.sender_output_width, 0, 1],
                [df_module.sender_output_width, df_module.sender_output_height, 1],
                [0, df_module.sender_output_height, 1],
            ], dtype=np.float32).T

            H_inv = np.linalg.inv(H)
            src_h = H_inv @ dst
            src = (src_h[:2, :] / src_h[2, :]).T.astype(int)

            cv2.polylines(display, [src.reshape(-1,1,2)],
                          True, (0,255,255), 3)

        except Exception as e:
            print("Could not draw rectangle:", e)

        if not found_and_saved:
            cv2.imwrite(SAVE_DETECTED_FRAME_AS, display)
            print("Saved detection frame as:", SAVE_DETECTED_FRAME_AS)
            found_and_saved = True

    cv2.imshow("Aruco Test", display)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if found_and_saved:
    print("At least one detection succeeded.")
else:
    print("No detections occurred in this video.")
