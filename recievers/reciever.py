# --- Imports ---
import cv2
import time
import numpy as np

# --- Definitions ---
binary_duration = 0.3   # seconds per bit
delimiter_duration = 0.5

# --- Setup capture ---
cap = cv2.VideoCapture(r"C:\my_projects\optical-laptop-communication\recievers\liläng_part2.mp4")
if not cap.isOpened():
    print("Error: Could not open camera/video.")
    exit()

# 🔹 NEW: get FPS from video
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback
frame_delay_ms = int(1000 / fps)  # delay per frame in milliseconds

# 🔹 NEW: record start time of the program
start_time = time.time()

# 🔹 NEW: detect if it's a video file
is_video_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0

# --- Helper function to read dominant color in ROI ---
def read_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV ranges
    red_lower1, red_upper1 = (0, 100, 100), (10, 255, 255)
    red_lower2, red_upper2 = (160, 100, 100), (179, 255, 255)
    white_lower, white_upper = (0, 0, 200), (180, 30, 255)
    black_lower, black_upper = (0, 0, 0), (180, 255, 50)
    green_lower, green_upper = (40, 50, 50), (80, 255, 255)

    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    black_mask = cv2.inRange(hsv, black_lower, black_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    counts = {
        "red": int(cv2.countNonZero(red_mask)),
        "white": int(cv2.countNonZero(white_mask)),
        "black": int(cv2.countNonZero(black_mask)),
        "green": int(cv2.countNonZero(green_mask)),
    }
    return max(counts, key=counts.get)

# --- Main function ---
def receive_message(bit_time=binary_duration, delim_time=delimiter_duration):
    bits = ""
    message = ""
    last_color = None
    waiting_for_sync = True
    decoding = False
    last_tick = None  # 🔹 track timing for bit windows

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

        # 🔹 Use video timestamp if it's a video file, otherwise wall clock
        if is_video_file:
            current_tick = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # seconds
        else:
            current_tick = time.time() - start_time  # seconds since program started

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = 100
        x1, x2 = max(0, cx-size), min(w, cx+size)
        y1, y2 = max(0, cy-size), min(h, cy+size)
        roi = frame[y1:y2, x1:x2]

        color = read_color(roi)

        # Visualization
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Receiver", frame)

        # --- SYNC logic ---
        if waiting_for_sync:
            if color == "green":
                if last_color != "green":
                    print("Green detected — syncing...")
                last_tick = None  # 🔹 reset timer while waiting
            elif color != "green" and last_color == "green":
                print("Green ended — starting decoding!")
                waiting_for_sync = False
                decoding = True
                bits = ""
                message = ""
                # 🔹 Initialize timer to current frame so first bit is counted immediately
                last_tick = current_tick - bit_time  
        elif decoding:
            # --- Handle delimiter first ---
            if color == "red" and color != last_color:
                while len(bits) >= 8:
                    byte = bits[:8]
                    bits = bits[8:]
                    try:
                        ch = chr(int(byte, 2))
                    except:
                        ch = '?'
                    message += ch
                    print(f"Received char: {ch}")
                # 🔹 pad remaining bits if < 8
                if 0 < len(bits) < 8:
                    byte = bits.ljust(8, '0')
                    try:
                        ch = chr(int(byte, 2))
                    except:
                        ch = '?'
                    message += ch
                    print(f"Received char (padded): {ch}")
                bits = ""
                last_tick = current_tick  # 🔹 reset timer after delimiter
                continue

            # --- Regular bit reading ---
            if last_tick is None:
                last_tick = current_tick

            elapsed = current_tick - last_tick
            if elapsed >= bit_time:
                if color == "white":
                    bits += "1"
                    print("Bit: 1")
                elif color == "black":
                    bits += "0"
                    print("Bit: 0")
                # 🔹 update last_tick immediately after reading a bit
                last_tick = current_tick

                last_color = color
                key = cv2.waitKey(frame_delay_ms) & 0xFF  # 🔹 use video FPS to delay frames
                if key == ord('q'):
                    break

    print("Final message:", message)
    cap.release()
    cv2.destroyAllWindows()

# --- Run ---
receive_message()
