# color_utils.py
import cv2
import numpy as np
from collections import Counter
from utilities.global_definitions import number_of_rows as rows, number_of_columns as cols


class BitColorTracker:
    def __init__(self):
        self.rows = rows
        self.cols = cols
        self.current_bit_roi = [[[] for _ in range(cols)] for _ in range(rows)]

    def add_frame(self, hsv_roi, row, col):
        self.current_bit_roi[row][col].append(hsv_roi)

    def end_bit(self, row, col):
        frames = self.current_bit_roi[row][col]
        if not frames:
            return None

        # classify each frame via LUT
        frame_classes = [
            classify_frame_LUT(f)
            for f in frames
        ]

        # majority vote
        frame_classes = np.array(frame_classes, dtype=np.int16)
        majority_idx = int(np.bincount(frame_classes).argmax())

        # reset buffer
        self.current_bit_roi[row][col] = []

        white_index = self.color_names.index("white")
        return "1" if majority_idx == white_index else "0"

    def reset(self):
        self.current_bit_roi = [[[] for _ in range(self.cols)] for _ in range(self.rows)]

    def colors(self, LUT, color_names):

        self.LUT = LUT
        self.color_names = color_names

colorTracker = BitColorTracker()

tracker = BitColorTracker()

# --- Compute a full HSV → COLOR lookup table (LUT) after corrected ranges ---

def build_color_LUT(corrected_ranges):
    """
    Build a 180 x 256 x 256 LUT mapping HSV -> class index.
    Class indices follow the order of corrected_ranges keys.
    """
    # Color index map
    color_names = list(corrected_ranges.keys())   # ["red1", "red2", "white", ...]
    num_colors = len(color_names)

    # Create empty LUT (180×256×256)
    LUT = np.zeros((180, 256, 256), dtype=np.uint8)

    # Prepare full HSV cube (180×256×256)
    H = np.arange(180)[:, None, None]
    S = np.arange(256)[None, :, None]
    V = np.arange(256)[None, None, :]

    # Broadcast to 3D grid
    H = H + np.zeros((1, 256, 256), dtype=np.uint16)
    S = np.zeros((180, 1, 256), dtype=np.uint16) + S
    V = np.zeros((180, 256, 1), dtype=np.uint16) + V

    # Fill LUT by writing integer class indices
    for idx, (color, (lower, upper)) in enumerate(corrected_ranges.items()):
        lh, ls, lv = lower
        uh, us, uv = upper

        if lh <= uh:
            # Normal range
            mask = (
                (H >= lh) & (H <= uh) &
                (S >= ls) & (S <= us) &
                (V >= lv) & (V <= uv)
            )
        else:
            # Hue wraps around (e.g. red)
            mask = (
                ((H >= lh) | (H <= uh)) &
                (S >= ls) & (S <= us) &
                (V >= lv) & (V <= uv)
            )

        LUT[mask] = idx

    return LUT, color_names

# --- Classifies the majority of the colors with help of LUT ---

def classify_frame_LUT(hsv):
    LUT=tracker.LUT

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # O(1) lookup for every pixel
    classes = LUT[H, S, V]

    # majority vote
    values, counts = np.unique(classes, return_counts=True)
    return int(values[counts.argmax()])



def dominant_color(hsv):
    LUT = tracker.LUT
    names = tracker.color_names

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    classes = LUT[H, S, V]
    hist = np.bincount(classes.ravel(), minlength=len(names))
    return names[int(hist.argmax())]

def color_offset_calculation(roi):

    """
    Calculates color offsets based on a calibration ROI containing red, green, and blue stripes.

    Arguments:
        "roi" (numpy.ndarray): The region of interest image in BGR format.
    
    Returns:
        "corrected_hsv_ranges" (dict): A dictionary containing the corrected HSV ranges for various colors.

    """

    expected_hsv_ranges = {
    "red": np.array([0, 255, 255]),
    "green": np.array([60, 255, 255]),
    "blue": np.array([120, 255, 255])
    }

    original_hsv_ranges = {
        "red1":  ([0, 100, 100], [10, 255, 255]),
        "red2":  ([160, 100, 100], [179, 255, 255]),
        "white": ([0, 0, 200], [179, 30, 255]),
        "black": ([0, 0, 0], [179, 255, 50]),
        "green": ([40, 50, 50], [80, 255, 255]),
        "blue":  ([100, 150, 0], [140, 255, 255])
    }

    def calculate_hue_difference(expected_hue_value, observed_hue_value):

        """
        Calculates the shortest difference between two hue values.

        Arguments:
            "expected_hue_value" (float): The expected hue value.
            "observed_hue_value" (float): The observed hue value.

        Returns:
            "hue_difference" (float): The shortest difference between the expected and observed hue values.

        """

        hue_difference = (expected_hue_value - observed_hue_value + 90) % 180 - 90

        return hue_difference
        
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(float) # Converts the ROI to HSV format
    
    stripe_width = roi.shape[1] // 3 # Width of each color stripe
    patch_width = int(stripe_width * 0.5) # Width of the patch to sample within each stripe
    start_offset = (stripe_width - patch_width) // 2 # Offset to center the patch within the stripe

    observed_hsv_dictionary = {}
    
    for stripe_index, color in enumerate(["red", "green", "blue"]):

        x_start = stripe_index * stripe_width + start_offset
        x_end = x_start + patch_width

        roi_stripe = roi_hsv[:, x_start:x_end]

        observed_hsv_dictionary[color] = np.median(roi_stripe.reshape(-1,3), axis = 0)
    
    hue_differences = []
    saturation_differences = []
    value_differences = []
    
    for color in ["red", "green", "blue"]:
        
        expected_hsv_range = expected_hsv_ranges[color].astype(float)
        observed_hsv = observed_hsv_dictionary[color].astype(float)

        hue_differences.append(calculate_hue_difference(float(expected_hsv_range[0]), float(observed_hsv[0])))  
        saturation_differences.append(float(expected_hsv_range[1]) - float(observed_hsv[1]))
        value_differences.append(float(expected_hsv_range[2]) - float(observed_hsv[2]))
    
    average_hue_offset = np.mean(hue_differences)
    average_saturation_offset = np.mean(saturation_differences)
    average_value_offset = np.mean(value_differences)

    print("\n=== Average HSV offsets applied ===")
    print(f"Average H offset: {average_hue_offset:.2f}")
    print(f"Average S offset: {average_saturation_offset:.2f}")
    print(f"Average V offset: {average_value_offset:.2f}\n")
    
    corrected_ranges = {}

    for color, (lower, upper) in original_hsv_ranges.items():

        lower = np.array(lower, dtype=float)
        upper = np.array(upper, dtype=float)

        lower_h = (lower[0] + average_hue_offset) % 180
        upper_h = (upper[0] + average_hue_offset) % 180

        sv_offset = np.array([average_saturation_offset, average_value_offset])

        lower_sv = np.clip(lower[1:] + sv_offset, 0, 255)
        upper_sv = np.clip(upper[1:] + sv_offset, 0, 255)

        lower_corrected = np.array([lower_h, lower_sv[0], lower_sv[1]])
        upper_corrected = np.array([upper_h, upper_sv[0], upper_sv[1]])

        lower_corrected = np.clip(lower_corrected, [0,0,0], [179,255,255]).astype(int)
        upper_corrected = np.clip(upper_corrected, [0,0,0], [179,255,255]).astype(int)

        corrected_ranges[color] = (lower_corrected, upper_corrected)
        
    return corrected_ranges

def dominant_color_with_offsets(roi, corrected_ranges):

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    masks = {}
    for color, (lower, upper) in corrected_ranges.items():
        lh = int(lower[0]); uh = int(upper[0])
        lower_arr = np.array(lower, dtype=np.int32)
        upper_arr = np.array(upper, dtype=np.int32)
        if lh <= uh:
            mask = cv2.inRange(hsv, lower_arr, upper_arr)
        else:
            # wrap: [lh..179] OR [0..uh]
            part1 = cv2.inRange(hsv, np.array([lh, lower_arr[1], lower_arr[2]]), np.array([179, upper_arr[1], upper_arr[2]]))
            part2 = cv2.inRange(hsv, np.array([0, lower_arr[1], lower_arr[2]]), np.array([uh, upper_arr[1], upper_arr[2]]))
            mask = part1 | part2
        masks[color] = mask

    counts = {c: int(cv2.countNonZero(m)) for c,m in masks.items()}
    if not counts:
        return None
    return max(counts, key=counts.get)

    
