# --- Imports ---

import cv2
import time
import numpy as np

# --- Definitions ---

binary_duration = 0.3
delimiter_duration = 0.5

# --- Setup camera ---

cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback if OpenCV can’t read it
delay = int(1000 / fps)

# --- Helper functions ---

def read_color(frame):
    """
    Detects the dominant color (black, white, red, green) in the center of the frame.
    Returns one of: 'black', 'white', 'red', 'green'
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges
    red_lower1, red_upper1 = (0, 100, 100), (10, 255, 255)
    red_lower2, red_upper2 = (160, 100, 100), (179, 255, 255)
    white_lower, white_upper = (0, 0, 200), (180, 30, 255)
    black_lower, black_upper = (0, 0, 0), (180, 255, 50)
    green_lower, green_upper = (40, 50, 50), (80, 255, 255)

    # Create masks
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    black_mask = cv2.inRange(hsv, black_lower, black_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Count pixels
    counts = {
        "red": cv2.countNonZero(red_mask),
        "white": cv2.countNonZero(white_mask),
        "black": cv2.countNonZero(black_mask),
        "green": cv2.countNonZero(green_mask)
    }

    # Pick dominant color
    return max(counts, key=counts.get)


# --- Main function ---

def receive_message(duration_per_bit=binary_duration, delimiter_time=delimiter_duration):
    bits = ""
    message = ""
    last_color = None
    decoding = False
    waiting_for_sync = True

    print("Receiver started — showing camera feed.")
    print("Waiting for GREEN signal to sync...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        size = 100
        roi = frame[cy-size:cy+size, cx-size:cx+size]

        color = read_color(roi)

        # Draw visualization
        cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Receiver", frame)

        # --- Logic for automatic start after green sync ---

        if waiting_for_sync:
            if color == "green":
                print("Green detected — syncing...")
            elif color != "green" and last_color == "green":
                print("Green ended — starting decoding!")
                waiting_for_sync = False
                decoding = True
                time.sleep(duration_per_bit)  # short stabilization pause

        elif decoding:
            if color in ["black", "white"] and color != last_color:
                bit = "1" if color == "white" else "0"
                bits += bit
                print(f"Bit: {bit}")
                time.sleep(duration_per_bit)

            elif color == "red" and color != last_color:
                if len(bits) == 8:
                    char = chr(int(bits, 2))
                    message += char
                    print(f"Received char: {char}")
                bits = ""
                time.sleep(delimiter_time)

        last_color = color
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    print("Final message:", message)
    cap.release()
    cv2.destroyAllWindows()


# --- Main execution ---
receive_message()
