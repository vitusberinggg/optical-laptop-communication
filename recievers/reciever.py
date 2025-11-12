# --- Imports ---
import cv2
import time
import numpy as np

# --- Config ---
binary_duration = 0.3
delimiter_duration = 0.5

# --- Camera setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
delay = int(1000 / fps)

# --- Color detection helper ---
def read_color(frame):
    """Detects dominant color (black, white, red, green) in center region."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV ranges for colors
    red_mask = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255)) | cv2.inRange(hsv, (160, 80, 50), (179, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 80, 50), (85, 255, 255))
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))

    counts = {
        "red": int(cv2.countNonZero(red_mask)),
        "green": int(cv2.countNonZero(green_mask)),
        "white": int(cv2.countNonZero(white_mask)),
        "black": int(cv2.countNonZero(black_mask)),
    }
    return max(counts, key=counts.get)

# --- Main function ---
def receive_message():
    bits = ""
    message = ""
    decoding = False
    last_color = None
    start_sync = False

    print("Receiver ready. Press 's' to start decoding, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading camera frame.")
            break

        # Center region to detect color
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        size = 120
        roi = frame[cy-size:cy+size, cx-size:cx+size]
        color = read_color(roi)

        # Visual feedback
        cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Receiver", frame)

        # --- Decode logic ---
        if decoding:
            if color == "green" and not start_sync:
                print("[SYNC] Starting message detection...")
                start_sync = True
                bits = ""
                message = ""
                time.sleep(binary_duration)
                continue

            if not start_sync:
                # ignore colors before sync
                last_color = color
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            if color in ["white", "black"] and color != last_color:
                bit = "1" if color == "white" else "0"
                bits += bit
                print(f"Bit: {bit}")
                time.sleep(binary_duration)

            elif color == "red" and color != last_color:
                if len(bits) == 8:
                    char = chr(int(bits, 2))
                    message += char
                    print(f"Received char: {char}")
                bits = ""
                time.sleep(delimiter_duration)

        last_color = color
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            decoding = True
            print("Decoding started...")
        elif key == ord('q'):
            break

    print("Final message:", message)
    cap.release()
    cv2.destroyAllWindows()

# --- Run ---
receive_message()
