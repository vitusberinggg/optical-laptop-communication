# --- Imports ---
import cv2
import time
import numpy as np

# --- Definitions ---
delimiter_duration = 0.5  # red duration
binary_duration = 0.3     # unused, just for reference

# --- Setup capture ---
cap = cv2.VideoCapture(r"C:\my_projects\optical-laptop-communication\recievers\liläng_part3.1.mp4")
if not cap.isOpened():
    print("Error: Could not open camera/video.")
    exit()

# 🔹 Get FPS and frame delay
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
frame_delay_ms = int(1000 / fps)

# 🔹 Record start time and detect if video file
start_time = time.time()
is_video_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0

# --- Helper: detect dominant color ---
def read_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, (0,100,100), (10,255,255)) | cv2.inRange(hsv, (160,100,100), (179,255,255))
    white_mask = cv2.inRange(hsv, (0,0,200), (180,30,255))
    black_mask = cv2.inRange(hsv, (0,0,0), (180,255,50))
    green_mask = cv2.inRange(hsv, (40,50,50), (80,255,255))
    blue_mask  = cv2.inRange(hsv, (100,150,0), (140,255,255))  # padding

    counts = {
        "red": int(cv2.countNonZero(red_mask)),
        "white": int(cv2.countNonZero(white_mask)),
        "black": int(cv2.countNonZero(black_mask)),
        "green": int(cv2.countNonZero(green_mask)),
        "blue": int(cv2.countNonZero(blue_mask)),
    }
    return max(counts, key=counts.get)

# --- Main function ---
def receive_message():
    bits = ""
    message = ""
    last_color = None
    waiting_for_sync = True
    decoding = False
    bit_ready = False  # set True when blue frame appears or first bit after green

    print("Receiver started — waiting for GREEN to sync...")
    print(f"is_video_file = {is_video_file}, fps={fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_video_file:
                print("End of video file reached.")
                break
            print("Error: Failed to capture frame.")
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        size = 100
        roi = frame[max(0,cy-size):min(h,cy+size), max(0,cx-size):min(w,cx+size)]

        color = read_color(roi)

        # Visualization
        cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), (0,255,0), 2)
        cv2.putText(frame, f"Detected: {color}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.imshow("Receiver", frame)

        # --- SYNC ---
        if waiting_for_sync:
            if color == "green" and last_color != "green":
                print("Green detected — syncing...")
            elif color != "green" and last_color == "green":
                print("Green ended — starting decoding!")
                waiting_for_sync = False
                decoding = True
                bit_ready = True  # 🔹 important: capture first bit immediately after green

        elif decoding:
            if color == "blue":
                bit_ready = True  # next non-blue frame is a valid bit
            elif color in ["white","black"] and bit_ready:
                bits += "1" if color == "white" else "0"
                print(f"Bit: {bits[-1]}")
                bit_ready = False  # wait for next blue before reading another
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
        key = cv2.waitKey(frame_delay_ms) & 0xFF
        if key == ord('q'):
            break

    print("Final message:", message)
    cap.release()
    cv2.destroyAllWindows()

# --- Run ---
receive_message()
