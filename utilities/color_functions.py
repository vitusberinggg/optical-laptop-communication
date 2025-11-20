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