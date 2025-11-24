# --- Imports ---
from recievers.webCamSim import VideoThreadedCapture
from recievers.color_utils import dominant_color, tracker  # 🔹 updated: import tracker
from utilities import detection_functions

import cv2
import time
import numpy as np

# --- Definitions ---
delimiter_duration = 0.5  # red duration
binary_duration = 0.3     # unused, just for reference

# --- Setup capture ---
cap = VideoThreadedCapture(r"C:\Users\ejadmax\code\optical-laptop-communication\recievers\gandalf2.0.mp4")
if not cap.isOpened():
    print("Error: Could not open camera/video.")
    exit()

#cv2.namedWindow("Receiver", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("Receiver", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("Receiver", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Receiver", 1920, 1200)

while True:
    ret, frame = cap.read()
    if ret:
        break
    time.sleep(0.01)  # small wait to avoid busy loop

# --- Main function ---
def receive_message():

    bits = ""
    message = ""
    last_color = None
    waiting_for_sync = True
    decoding = False
    current_bit_colors = []  # DEBUG: store colors per bit

    homography_matrix = None
    homography_matrix = None
    sender = None
    sender_output_width = None
    sender_output_height = None

    print("Receiver started — waiting for GREEN to sync...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        # Area that is being processed (roi)
        frame = cv2.flip(frame, 1)
        
        homography_matrix = detection_functions.detect_aruco_marker_frame(frame)
        if homography is not None:
            warped = cv2.warpPerspective(
                frame,
                homography,
                (sender_output_width, sender_output_height)
            )
            roi = warped   # <-- use warped screen as the ROI
        else:
            roi = frame    # fallback while still syncing

        color = dominant_color(roi)

        # --- DEBUG: show ROI info and color ---
        #print(f"ROI shape: {roi.shape}, Dominant color: {color}")

        # --- Visualization ---
        frame_with_roi = frame.copy()
        if homography is not None:
            # Sender screen corners in warped space
            h, w = sender_output_height, sender_output_width
            corners = np.array([
                [0, 0],         # top-left
                [w, 0],         # top-right
                [w, h],         # bottom-right
                [0, h]          # bottom-left
            ], dtype=np.float32).reshape(-1, 1, 2)

            # Map corners back to original frame
            projected_corners = cv2.perspectiveTransform(corners, np.linalg.inv(homography))

            # Draw polygon on original frame
            cv2.polylines(frame_with_roi, [np.int32(projected_corners)], True, (203, 192, 255), 3)  # pink polygon

        # Show the original frame with polygon
        #cv2.imshow("Original Frame with Computer ROI", frame_with_roi)

        cv2.imshow("Receiver", roi)

        # --- SYNC ---
        if waiting_for_sync:
            if color == "green" and last_color != "green":
                print("Green detected — syncing...")
                tracker.reset()
            elif color != "green" and last_color == "green":
                print("Green ended — starting decoding!")
                tracker.reset()
                waiting_for_sync = False
                decoding = True

        elif decoding:
            if color == "blue" and last_color != "blue":
                # end of bit → compute average color
                majority_color = tracker.end_bit()
                #print(f"sender height: {sender_output_height}")
                #print(f"sender width: {sender_output_width}")

                if majority_color == "white":
                    bits += "1"
                elif majority_color == "black":
                    bits += "0"
                
                #print(f"color list: {current_bit_colors}") # DEBUGGING
                #current_bit_colors = [] # DEBUGGING
                #print(f"Bit: {bits[-1]} (averaged color = {majority_color})")

            elif color in ["white", "black"]:
                # part of bit → collect frame
                tracker.add_frame(roi)
                #current_bit_colors.append(color)
            elif color == "red" and last_color != "red":
                # delimiter: process accumulated bits as character(s)
                while len(bits) >= 8:
                    byte = bits[:8]
                    bits = bits[8:]
                    try:
                        ch = chr(int(byte,2))
                    except:
                        ch = '?'
                    message += ch
                    print(f"Received char: {ch}")
                if 0 < len(bits) < 8:
                    byte = bits.ljust(8,'0')  # pad incomplete bits
                    try:
                        ch = chr(int(byte,2))
                    except:
                        ch = '?'
                    message += ch
                    print(f"Received char (padded): {ch}")
                bits = ""

        last_color = color
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # DEBUG: show any remaining colors and bits
    if current_bit_colors:
        print(f"Colors collected for last unfinished bit: {current_bit_colors}")
    if bits:
        print(f"Remaining bits not yet converted: {bits}")

    print("Final message:", message)
    cap.release()
    cv2.destroyAllWindows()

# --- Run ---
receive_message()
