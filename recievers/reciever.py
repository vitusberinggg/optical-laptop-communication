# --- Imports ---
import cv2
import time
import numpy as np

# --- Config ---
binary_duration = 0.3
delimiter_duration = 0.5

# --- Camera setup ---
cap = cv2.VideoCapture(0)  # Change to video path if needed
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

    # HSV ranges for colors (tuned for screen light)
    red_mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) | cv2.inRange(hsv, (160, 70, 50), (179, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 50, 50), (85, 255, 255))
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))

    counts = {
        "red": cv2.countNonZero(red_mask),
        "green": cv2.countNonZero(green_mask),
        "white": cv2.countNonZero(white_mask),
        "black": cv2.countNonZero(black_mask),
    }

    # Return color with max detected pixels
    return max(counts, key=counts.get)

# --- Main function ---
def receive_message():
    bits = ""
    message = ""
    decoding = False
    preparing = False
    last_color = None

    print("Receiver ready. Waiting for green sync signal...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or camera read error.")
            break

        # Detect color in center region
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        size = 100
        roi = frame[cy-size:cy+size, cx-size:cx+size]
        color = read_color(roi)

        # Draw detection region and label
        cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {color}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Receiver", frame)

        # --- Logic flow ---
        if not preparing and color == "green":
            # Green detected: prepare to start soon
            preparing = True
            print("[SYNC] Green detected — preparing for decoding...")
            bits = ""
            message = ""

        elif preparing and color != "green" and not decoding:
            # Transition from green → something else = start decoding
            decoding = True
            preparing = False
            print("[SYNC] Transition from green detected — decoding started!")

        elif decoding:
            # Decode bits and characters
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

        # Exit if user presses Q
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    print("\nFinal message:", message)
    cap.release()
    cv2.destroyAllWindows()

# --- Run ---
receive_message()
