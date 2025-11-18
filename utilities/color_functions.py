# color_utils.py
import cv2
import numpy as np
from collections import Counter

class BitColorTracker:
    def __init__(self):
        self.current_bit_roi = [[[]]]  # stores all the ROIs

    def add_frame(self, roi, row=0, col=0):
        """Add a new ROI frame for the current bit."""
        # Expand rows if necessary
        while len(self.current_bit_roi) <= row:
            self.current_bit_roi.append([])

        # Expand columns for this row if necessary
        while len(self.current_bit_roi[row]) <= col:
            self.current_bit_roi[row].append([])

        self.current_bit_roi[row][col].append(roi)

    def end_bit(self, row=0, col=0):
        """Compute the dominant color for the bit and reset frame buffer."""
        if not self.current_bit_roi:
            return None

        # Convert each frame to HSV and get the dominant color per frame
        frame_colors = []
        for frame in self.current_bit_roi[row][col]:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # RED — unchanged (red works well already)
            red_mask = cv2.inRange(hsv, (0,120,120), (10,255,255)) | \
                    cv2.inRange(hsv, (160,120,120), (179,255,255))

            # WHITE — tighten saturation to avoid confusion with light blue
            white_mask = cv2.inRange(hsv, (0,0,220), (180,25,255))

            # BLACK — much stricter brightness limit (prevents dark blue being classified as black)
            black_mask = cv2.inRange(hsv, (0,0,0), (180,255,35))

            # GREEN — narrowed to avoid overlap with blue
            green_mask = cv2.inRange(hsv, (45,80,80), (75,255,255))

            # BLUE — shifted upward to avoid dark/black confusion
            blue_mask  = cv2.inRange(hsv, (95,120,70), (130,255,255))

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
        self.current_bit_roi = [[[]]]
        return majority_color

    def reset(self):
        self.current_bit_roi = [[[]]]

# For backward compatibility
tracker = BitColorTracker()
def dominant_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, (0,100,100), (10,255,255)) | \
               cv2.inRange(hsv,(160,100,100),(179,255,255))
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