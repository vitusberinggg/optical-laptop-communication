import cv2
import numpy as np

# --- Define HSV ranges for colors ---
COLOR_RANGES = {
    "red": [((0, 100, 100), (10, 255, 255)),
            ((160, 100, 100), (179, 255, 255))],
    "white": [((0, 0, 200), (180, 30, 255))],
    "black": [((0, 0, 0), (180, 255, 50))],
    "green": [((40, 50, 50), (80, 255, 255))],
    "blue": [((100, 150, 0), (140, 255, 255))],
}

def dominant_color(frame):
    """
    Returns the color with the most pixels in the frame.

    Arguments:
        frame: BGR image (numpy array)
    
    Returns:
        color_name: str
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    counts = {}

    for color_name, ranges in COLOR_RANGES.items():
        mask_total = None
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if mask_total is None:
                mask_total = mask
            else:
                mask_total = cv2.bitwise_or(mask_total, mask)
        counts[color_name] = cv2.countNonZero(mask_total)
    
    # Return the color with the highest count
    return max(counts, key=counts.get)
