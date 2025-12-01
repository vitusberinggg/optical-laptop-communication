
# ---- Imports ---

import cv2
from collections import Counter

from utilities.global_definitions import (
    number_of_rows, number_of_columns,
    red_lower_hsv_limit_1, red_upper_hsv_limit_1,
    red_lower_hsv_limit_2, red_upper_hsv_limit_2,
    white_lower_hsv_limit, white_upper_hsv_limit,
    black_lower_hsv_limit, black_upper_hsv_limit,
    green_lower_hsv_limit, green_upper_hsv_limit,
    blue_lower_hsv_limit, blue_upper_hsv_limit
)

# --- Functions ---

class BitColorTracker:

    def __init__(self):

        """

        """

        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns
        self.current_bit_roi = [[[] for _ in range(number_of_columns)] for _ in range(number_of_rows)]
        self.patch_fraction = 0.3

    def add_frame(self, roi, row, col):
        
        """
        
        """

        #self.current_bit_roi[row][col].append(roi)
        
        h, w = roi.shape[:2]
        patch_h = int(h * self.patch_fraction)
        patch_w = int(w * self.patch_fraction)

        start_y = (h - patch_h) // 2
        start_x = (w - patch_w) // 2

        patch = roi[start_y:start_y+patch_h, start_x:start_x+patch_w]
        self.current_bit_roi[row][col].append(patch)
        

    def end_bit(self, row, col):
        
        """
        
        """

        frames = self.current_bit_roi[row][col]

        if len(frames) == 0:
            return None

        frame_colors = []

        for frame in frames:
            color = dominant_color(frame)
            frame_colors.append(color)

        majority = Counter(frame_colors).most_common(1)[0][0]

        self.current_bit_roi[row][col] = []

        if color == "black":
            return "0"
        
        else:
            return "1"

    def reset(self):

        """
        
        """

        self.current_bit_roi = [[[] for _ in range(self.number_of_columns)] for _ in range(self.number_of_rows)]

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

    red_mask = cv2.inRange(hsv, red_lower_hsv_limit_1, red_upper_hsv_limit_1) | cv2.inRange(hsv, red_lower_hsv_limit_2, red_upper_hsv_limit_2)

    white_mask = cv2.inRange(hsv, white_lower_hsv_limit, white_upper_hsv_limit)

    black_mask = cv2.inRange(hsv, black_lower_hsv_limit, black_upper_hsv_limit)

    green_mask = cv2.inRange(hsv, green_lower_hsv_limit, green_upper_hsv_limit)

    blue_mask  = cv2.inRange(hsv, blue_lower_hsv_limit, blue_upper_hsv_limit)

    counts = {
        "red": int(cv2.countNonZero(red_mask)),
        "white": int(cv2.countNonZero(white_mask)),
        "black": int(cv2.countNonZero(black_mask)),
        "green": int(cv2.countNonZero(green_mask)),
        "blue": int(cv2.countNonZero(blue_mask)),
    }

    return max(counts, key = counts.get)
