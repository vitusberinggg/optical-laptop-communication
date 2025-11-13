# --- Imports ---
import cv2
import numpy as np

# --- Function to decode bitstream with blue separators ---
def decode_bits_with_blue(frames, roi_size=100, verbose=False):
    """
    Decodes bits from a sequence of frames where each bit is a colored frame:
    - white: bit 1
    - black: bit 0
    - blue: separator (next bit ready)
    
    Arguments:
        frames (list of np.ndarray): List of frames (BGR) from the sender.
        roi_size (int): Size of the square region to read the color from.
        verbose (bool): Print debug info.
    
    Returns:
        str: Binary string representing the decoded bits.
    """
    def read_color(frame):
        """Detects dominant color in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0,0,200), (180,30,255))
        black_mask = cv2.inRange(hsv, (0,0,0), (180,255,50))
        blue_mask  = cv2.inRange(hsv, (100,150,0), (140,255,255))
        counts = {
            "white": int(cv2.countNonZero(white_mask)),
            "black": int(cv2.countNonZero(black_mask)),
            "blue": int(cv2.countNonZero(blue_mask)),
        }
        return max(counts, key=counts.get)

    bits = ""
    last_color = None
    bit_ready = True  # ready for first bit

    for frame in frames:
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        roi = frame[max(0,cy-roi_size):min(h,cy+roi_size), max(0,cx-roi_size):min(w,cx+roi_size)]
        color = read_color(roi)

        if color == "blue":
            bit_ready = True
            if verbose:
                print("Blue separator detected — next bit ready.")
        elif color in ["white","black"] and bit_ready:
            bits += "1" if color == "white" else "0"
            if verbose:
                print(f"Decoded bit: {bits[-1]}")
            bit_ready = False  # wait for next blue separator

        last_color = color

    return bits
