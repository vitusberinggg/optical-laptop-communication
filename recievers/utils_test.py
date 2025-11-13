# color_utils.py
import cv2
import numpy as np
from collections import Counter

class BitColorTracker:
    def __init__(self):
        self.current_bit_frames = []  # store all ROIs for the current bit
        self.bit_colors = []          # store final dominant colors for each bit

    def add_frame(self, roi):
        """Add a new ROI frame for the current bit."""
        self.current_bit_frames.append(roi)

    def end_bit(self):
        """Compute the dominant color for the bit and reset frame buffer."""
        if not self.current_bit_frames:
            return None

        # Convert each frame to HSV and get the dominant color per frame
        frame_colors = []
        for frame in self.current_bit_frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_mask = cv2.inRange(hsv, (0,100,100), (10,255,255)) | cv2.inRange(hsv, (160,100,100), (179,255,255))
            white_mask = cv2.inRange(hsv, (0,0,200), (180,30,255))
            black_mask = cv2.inRange(hsv, (0,0,0), (180,255,50))
            green_mask = cv2.inRange(hsv, (40,50,50), (80,255,255))
            blue_mask  = cv2.inRange(hsv, (100,150,0), (140,255,255))

            counts = {
                "red": int(cv2.countNonZero(red_mask)),
                "white": int(cv2.countNonZero(white_mask)),
                "black": int(cv2.countNonZero(black_mask)),
                "green": int(cv2.countNonZero(green_mask)),
                "blue": int(cv2.countNonZero(blue_mask)),
            }
            dominant = max(counts, key=counts.get)
            frame_colors.append(dominant)

        # Majority vote for this bit
        majority_color = Counter(frame_colors).most_common(1)[0][0]
        self.bit_colors.append(majority_color)
        self.current_bit_frames = []
        return majority_color

    def reset(self):
        self.current_bit_frames = []
        self.bit_colors = []

# For backward compatibility (optional)
tracker = BitColorTracker()
def dominant_color(roi):
    tracker.add_frame(roi)
    # Return current frame’s dominant as a temporary measure
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0,100,100), (10,255,255)) | cv2.inRange(hsv, (160,100,100), (179,255,255))
    white_mask = cv2.inRange(hsv, (0,0,200), (180,30,255))
    black_mask = cv2.inRange(hsv, (0,0,0), (180,255,50))
    green_mask = cv2.inRange(hsv, (40,50,50), (80,255,255))
    blue_mask  = cv2.inRange(hsv, (100,150,0), (140,255,255))
    counts = {
        "red": int(cv2.countNonZero(red_mask)),
        "white": int(cv2.countNonZero(white_mask)),
        "black": int(cv2.countNonZero(black_mask)),
        "green": int(cv2.countNonZero(green_mask)),
        "blue": int(cv2.countNonZero(blue_mask)),
    }
    return max(counts, key=counts.get)
