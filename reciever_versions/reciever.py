
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

# --- Helper functions ---

def read_color(frame):
    """
    Detects the dominant color (black, white, red) in the center of the frame.
    Returns one of: 'black', 'white', 'red'

    Arguments:
        frame : numpy.ndarray
            The current video frame captured from the camera (BGR format).

    Returns: 
        str
            A string indicating the dominant color detected: "black", "white", or "red".

    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges
    red_lower1 = (0, 100, 100)
    red_upper1 = (10, 255, 255)
    red_lower2 = (160, 100, 100)
    red_upper2 = (179, 255, 255)
    white_lower = (0, 0, 200)
    white_upper = (180, 30, 255)
    black_lower = (0, 0, 0)
    black_upper = (180, 255, 50)

    # Create masks
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    black_mask = cv2.inRange(hsv, black_lower, black_upper)

    # Count pixels
    red_count = cv2.countNonZero(red_mask)
    white_count = cv2.countNonZero(white_mask)
    black_count = cv2.countNonZero(black_mask)

    # Pick dominant color
    if red_count > white_count and red_count > black_count:
        return "red"
    elif white_count > black_count:
        return "white"
    else:
        return "black"


# --- Main function ---
def receive_message(duration_per_bit=binary_duration, delimiter_time=delimiter_duration):
    """ 
    Captures video from the camera and decodes a binary message based on detected screen colors.

    Arguments:
        duration_per_bit : float, optional
            Duration in seconds that represents one bit (default = binary_duration).
        delimiter_time : float, optional
            Duration in seconds used to detect character delimiters (default = delimiter_duration).

    Returns:
        None
            Displays live video and prints the decoded message to the console.

    """
    bits = ""
    message = ""
    last_color = None

    print("Receiver started — showing camera feed. Press 's' to start decoding or 'q' to quit.")

    decoding = False  # Only start decoding when user presses 's'

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip for easier viewing (optional)
        frame = cv2.flip(frame, 1)

        # Draw a center rectangle region (for sampling)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        size = 100
        roi = frame[cy-size:cy+size, cx-size:cx+size]

        color = read_color(roi)

        # Draw visualization
        cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Show camera feed
        cv2.imshow("Receiver", frame)

        # If decoding is active, process bits
        if decoding:
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

        if key == ord('s'):
            decoding = True
            print("Started decoding...")

        elif key == ord('q'):
            break

    print("Final message:", message)
    cap.release()
    cv2.destroyAllWindows()


# --- Main execution ---
receive_message()
