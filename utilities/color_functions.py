# color_utils.py
import cv2
import numpy as np
from collections import Counter
from utilities.global_definitions import number_of_rows as rows, number_of_columns as cols

class BitColorTracker:
    def __init__(self):
        # frame buffer for each bit (2D array)
        self.rows = rows
        self.cols = cols
        self.current_bit_roi = [[[] for _ in range(cols)] for _ in range(rows)]

    def add_frame(self, roi, row, col):
        self.current_bit_roi[row][col].append(roi)

    def end_bit(self, row, col):
        frames = self.current_bit_roi[row][col]
        if len(frames) == 0:
            return None

        frame_colors = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            red_mask = cv2.inRange(hsv, (0,120,120), (10,255,255)) | \
                       cv2.inRange(hsv, (160,120,120), (180,255,255))
            white_mask = cv2.inRange(hsv, (0,0,220), (180,25,255))
            black_mask = cv2.inRange(hsv, (0,0,0), (180,255,35))
            green_mask = cv2.inRange(hsv, (45,80,80), (75,255,255))
            blue_mask  = cv2.inRange(hsv, (95,120,70), (130,255,255))

            counts = {
                "red": int(cv2.countNonZero(red_mask)),
                "white": int(cv2.countNonZero(white_mask)),
                "black": int(cv2.countNonZero(black_mask)),
                "green": int(cv2.countNonZero(green_mask)),
                "blue": int(cv2.countNonZero(blue_mask)),
            }

            frame_colors.append(max(counts, key=counts.get))

        majority = Counter(frame_colors).most_common(1)[0][0]

        # reset ONLY this one bit
        self.current_bit_roi[row][col] = []

        # return integer bit
        return "1" if majority == "white" else "0"

    def reset(self):
        self.current_bit_roi = [[[] for _ in range(self.cols)] for _ in range(self.rows)]

# For backward compatibility
tracker = BitColorTracker()
def dominant_color(roi):
    """
    Computes the dominant color in the given ROI using HSV color space.

    Arguments:
        roi: The region of interest (ROI) frame to analyze.

    Returns:
        The dominant color as a string (e.g., "red", "white", etc.).
    
    """
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