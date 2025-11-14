# --- Imports ---
from webCamSim import VideoThreadedCapture
from color_utils import dominant_color, tracker  # 🔹 updated: import tracker

import cv2
import time
import numpy as np

# --- Definitions ---
delimiter_duration = 0.5  # red duration
binary_duration = 0.3     # unused, just for reference

# --- Setup capture ---
cap = VideoThreadedCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera/video.")
    exit()

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

    print("Receiver started — waiting for GREEN to sync...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        # Area that is being processed (roi)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        size = 100   
        roi = frame[max(0,cy-size):min(h,cy+size), max(0,cx-size):min(w,cx+size)]

        color = dominant_color(roi)

        # --- DEBUG: show ROI info and color ---
        #print(f"ROI shape: {roi.shape}, Dominant color: {color}")

        # Visualization
        cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), (0,255,0), 2)
        cv2.putText(frame, f"Detected: {color}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.imshow("Receiver", frame)

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

                if majority_color == "white":
                    bits += "1"
                elif majority_color == "black":
                    bits += "0"
                
                #print(f"color list: {current_bit_colors}") # DEBUGGING
                #current_bit_colors = [] # DEBUGGING
                print(f"Bit: {bits[-1]} (averaged color = {majority_color})")

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
