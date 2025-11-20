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
        if not frames:
            return None

        # Predefine HSV thresholds for all colors
        # (kept identical to your logic, but vectorized)
        def classify_frame(frame):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            H = hsv[:, :, 0]
            S = hsv[:, :, 1]
            V = hsv[:, :, 2]

            # Color masks (vectorized)
            red_mask = (((H <= 10) | (H >= 160)) &
                        (S >= 120) & (V >= 120))

            white_mask = (V >= 220) & (S <= 25)
            black_mask = (V <= 35)
            green_mask = (45 <= H) & (H <= 75) & (S >= 80) & (V >= 80)
            blue_mask  = (95 <= H) & (H <= 130) & (S >= 120) & (V >= 70)

            # Stack masks: shape (5, height, width)
            mask_stack = np.stack([
                red_mask,
                white_mask,
                black_mask,
                green_mask,
                blue_mask,
            ], axis=0)

            # Count all at once: vectorized
            counts = mask_stack.reshape(5, -1).sum(axis=1)

            return np.argmax(counts)  # returns int 0–4

        # Vectorized across all frames
        frame_colors = np.fromiter(
            (classify_frame(f) for f in frames),
            dtype=np.int8
        )

        # Majority vote on integers
        majority_color = int(np.bincount(frame_colors).argmax())

        # Reset bit buffer
        self.current_bit_roi[row][col] = []

        # Only white = "1", everything else = "0" (your original rule)
        return "1" if majority_color == 1 else "0"

    def reset(self):
        self.current_bit_roi = [[[] for _ in range(self.cols)] for _ in range(self.rows)]

# For backward compatibility
tracker = BitColorTracker()

def dominant_color(roi):
    """
    Fast dominant color detection optimized for Python 3.13 + OpenCV + NumPy.
    """
    if roi is None or roi.size == 0:
        return "black"

    # Convert to HSV once
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Boolean masks (vectorized)
    red_mask = (
        ((h <= 10) | (h >= 160)) &
        (s >= 100) & (v >= 100)
    )

    green_mask = (
        (h >= 40) & (h <= 80) &
        (s >= 50) & (v >= 50)
    )

    blue_mask = (
        (h >= 100) & (h <= 140) &
        (s >= 120)
    )

    white_mask = (v >= 200) & (s <= 30)
    black_mask = (v <= 40)

    # Count pixels (fast C implementation)
    counts = {
        "red": int(red_mask.sum()),
        "green": int(green_mask.sum()),
        "blue": int(blue_mask.sum()),
        "white": int(white_mask.sum()),
        "black": int(black_mask.sum()),
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

    
