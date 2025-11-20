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

def color_offset_calculation(color_reference_frame):

    expected_hsv = {
    "red": np.array([0, 255, 255]),
    "green": np.array([60, 255, 255]),
    "blue": np.array([120, 255, 255])
    }

    original_hsv_ranges = {
        "red1":  ([0, 100, 100], [10, 255, 255]),
        "red2":  ([160, 100, 100], [179, 255, 255]),
        "white": ([0, 0, 200], [180, 30, 255]),
        "black": ([0, 0, 0], [180, 255, 50]),
        "green": ([40, 50, 50], [80, 255, 255]),
        "blue":  ([100, 150, 0], [140, 255, 255])
    }

    def hue_diff(expected_h, observed_h):
        diff = (expected_h - observed_h + 90) % 180 - 90
        return diff
        
    calibrate_hsv = cv2.cvtColor(color_reference_frame, cv2.COLOR_BGR2HSV)
    
    stripe_width = color_reference_frame.shape[1] // 3
    observed_hsv = {}
    
    for i, color_name in enumerate(["red", "green", "blue"]):
        x_start = i * stripe_width
        x_end = (i + 1) * stripe_width
        roi = calibrate_hsv[:, x_start:x_end]
        observed_hsv[color_name] = np.median(roi.reshape(-1,3), axis=0)
    
    h_offsets = {}
    s_offsets = {}
    v_offsets = {}
    
    for color in ["red","green","blue"]:
        exp = expected_hsv[color].astype(float)
        obs = observed_hsv[color].astype(float)
        h_diffs.append(_hue_diff(float(exp[0]), float(obs[0])))  
        s_diffs.append(float(exp[1]) - float(obs[1]))
        v_diffs.append(float(exp[2]) - float(obs[2]))
    
    avg_h_offset = np.mean([h_offsets[c] for c in h_offsets])
    avg_s_offset = np.mean([s_offsets[c] for c in s_offsets])
    avg_v_offset = np.mean([v_offsets[c] for c in v_offsets])
    
    corrected_ranges = {}
    for color, (lower, upper) in original_hsv_ranges.items():
        lower_corrected = np.clip(np.array(lower) + np.array([avg_h_offset, avg_s_offset, avg_v_offset]), [0,0,0], [179,255,255])
        upper_corrected = np.clip(np.array(upper) + np.array([avg_h_offset, avg_s_offset, avg_v_offset]), [0,0,0], [179,255,255])
        corrected_ranges[color] = (lower_corrected.astype(int), upper_corrected.astype(int))
    return corrected_ranges

def dominant_color_with_offsets(roi, corrected_ranges):

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    masks = {
        color: cv2.inRange(hsv, lower, upper)
        for color, (lower, upper) in corrected_ranges.items()
    }
    counts = {color: int(cv2.countNonZero(mask)) for color, mask in masks.items()}

    return max(counts, key=counts.get)

    
