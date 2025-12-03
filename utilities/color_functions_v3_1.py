
# --- Imports ---

import time
import cv2
import numpy as np
from collections import Counter
from numba import njit, prange
from utilities.global_definitions import (
    number_of_rows as rows, number_of_columns as cols
    ,red_lower_hsv_limit_1, red_upper_hsv_limit_1,
    red_lower_hsv_limit_2, red_upper_hsv_limit_2,
    white_lower_hsv_limit, white_upper_hsv_limit,
    black_lower_hsv_limit, black_upper_hsv_limit,
    green_lower_hsv_limit, green_upper_hsv_limit,
    blue_lower_hsv_limit, blue_upper_hsv_limit,
    end_bit_steps, dominant_color_steps
    )

# --- Functions ---

class BitColorTracker:
    def __init__(self):
        self.time_in = 0
        self.rows = rows
        self.cols = cols
        self.hsv_frames = []
        self.LUT = None
        self.color_names = None

    def add_frame(self, hsv_roi):
        self.hsv_frames.append(hsv_roi)
        if not hasattr(BitColorTracker.add_frame, "time"):
            self.time_in = time.time()
            BitColorTracker.add_frame.time = ("chingchong")

        if not hasattr(BitColorTracker.add_frame, "walla") and (time.time() - self.time_in > 0.15) and self.time_in != 0:
            # debug dump
            
            # quick stats
            H = hsv_roi[:,:,0]; S = hsv_roi[:,:,1]; V = hsv_roi[:,:,2]
            print("HSV roi shape:", hsv_roi.shape, "dtype:", hsv_roi.dtype)
            print("H min/max:", int(H.min()), int(H.max()))
            print("S min/max:", int(S.min()), int(S.max()))
            print("V min/max:", int(V.min()), int(V.max()))
            BitColorTracker.add_frame.walla = ("bingbing")


    def _pad_frames(self, frames):
        # H = frame height, W = frame width
        _, H, W, _ = frames.shape

        # dividing H/W into cell_H/-W
        # always rounds up, ex. 4.1 -> 5
        cell_h = int(np.ceil(H / self.rows))
        cell_w = int(np.ceil(W / self.cols))

        # creating the padded H/W
        padded_H = cell_h * self.rows
        padded_W = cell_w * self.cols

        # calculating how much it should extend in H/W
        pad_bottom = padded_H - H
        pad_right = padded_W - W

        # making the padded frames
        padded_frames = np.pad(
            frames,
            ((0, 0), (0, pad_bottom), (0, pad_right), (0, 0)),
            mode='edge'
        )
        return padded_frames, cell_h, cell_w
  
    def end_bit(self):


        if len(self.hsv_frames) == 0:
            return None

        hsv_frames = np.asarray(self.hsv_frames)
        self.hsv_frames = []

        padded_frames, self.cell_h, self.cell_w = self._pad_frames(hsv_frames)

        N, H, W, _ = padded_frames.shape

        # Split into cells
        cells = padded_frames.reshape(N, self.rows, self.cell_h, self.cols, self.cell_w, 3)

        # --- Centered rectangle sampling inside each cell ---
        patch_h = max(self.cell_h // 2, 1)
        patch_w = max(self.cell_w // 2, 1)

        h0 = (self.cell_h - patch_h) // 2
        h1 = h0 + patch_h
        w0 = (self.cell_w - patch_w) // 2
        w1 = w0 + patch_w

        sampled_cells = cells[:, :, h0:h1, :, w0:w1, :]  # shape: (N, rows, patch_h, cols, patch_w, 3)
        if patch_h > end_bit_steps and patch_w > end_bit_steps:
            sampled_cells = sampled_cells[:, :, ::end_bit_steps, :, ::end_bit_steps, :] # adding pixel steps to prevent too many pixels  

        # HSV → class IDs using LUT
        Hc = sampled_cells[..., 0].astype(np.uint16)
        Sc = sampled_cells[..., 1].astype(np.uint16)
        Vc = sampled_cells[..., 2].astype(np.uint16)
        classes = self.LUT[Hc, Sc, Vc]  # shape: (N, rows, patch_h, cols, patch_w)

        # Merge all samples per cell
        merged = classes  # shape: (N, rows, patch_h, cols, patch_w)

        # Majority vote
        num_classes = int(self.LUT.max()) + 1
        bitgrid = bitgrid_majority_calc(merged, num_classes)

        #print(f"[DEBUG] bitgrid with color ids: \n{bitgrid}")

        white_idx = 2
        bitgrid_str = np.where(bitgrid == white_idx, "1", "0")

        #print(f"[DEBUG] bitgrid: \n{bitgrid_str}")

        return bitgrid_str


    def reset(self):
        self.hsv_frames = []

    def colors(self, LUT, color_names):
        self.LUT = LUT
        self.color_names = color_names


@njit(parallel=True)
def bitgrid_majority_calc(merged, num_classes):
    # merged shape: (N, rows, patch_h, cols, patch_w)
    N, R, Ph, C, Pw = merged.shape

    out = np.empty((R, C), dtype=np.int32)

    # parallell looping for speed
    for r in prange(R):
        for c in range(C):

            # histogram will help us calc the dominant color
            # histogram for this cell
            counts = np.zeros(num_classes, dtype=np.int32)

            # accumulate across all frames and samples inside the cell
            for n in range(N):
                for ph in range(Ph):
                    for pw in range(Pw):
                        cls = merged[n, r, ph, c, pw]
                        if 0 <= cls < num_classes:
                            counts[cls] += 1

            # find the average of this particular cell
            max_class = 0
            max_val = counts[0]
            for k in range(1, num_classes):
                if counts[k] > max_val:
                    max_class = k
                    max_val = counts[k]

            # gives the average of the cell to that particular position
            # gives the value is the "id" of that particular color
            out[r, c] = max_class

    # returns an 2-D array of color ids
    return out


tracker = BitColorTracker()

# --- Compute a full HSV → COLOR lookup table (LUT) after corrected ranges ---
# corrected ranges need to be in np.array for it to classify color ids properly

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
    H = np.broadcast_to(np.arange(180, dtype=np.uint16)[:,None,None], (180,256,256))
    S = np.broadcast_to(np.arange(256, dtype=np.uint16)[None,:,None], (180,256,256))
    V = np.broadcast_to(np.arange(256, dtype=np.uint16)[None,None,:], (180,256,256))

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


def dominant_color_hsv(hsv):

    LUT = tracker.LUT
    names = tracker.color_names
    
    ph, pw, _ = hsv.shape
    
    if ph > dominant_color_steps and pw > dominant_color_steps: # adding pixel steps to prevent too many pixels
        hsv = hsv [::dominant_color_steps, ::dominant_color_steps, :]

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    classes = LUT[H, S, V]

    hist = np.bincount(classes.ravel(), minlength=len(names))

    names = names[int(hist.argmax())]

    if names == "red1" or names == "red2":
        return "red"
    else:
        return names

def dominant_color_bgr(roi):

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

def color_offset_calculation(roi):

    """
    Calculates color offsets based on a calibration ROI containing red, green, and blue stripes.

    Arguments:
        "roi" (numpy.ndarray): The region of interest image in BGR format.
    
    Returns:
        "corrected_hsv_ranges" (dict): A dictionary containing the corrected HSV ranges for various colors.

    """

    expected_hsv_ranges = {
    "blue": np.array([120, 255, 255]),
    "green": np.array([60, 255, 255]),
     "red": np.array([0, 255, 255])
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
    
    for stripe_index, color in enumerate(["blue", "green", "red"]):

        x_start = stripe_index * stripe_width + start_offset
        x_end = x_start + patch_width

        roi_stripe = roi_hsv[:, x_start:x_end]

        observed_hsv_dictionary[color] = np.median(roi_stripe.reshape(-1,3), axis = 0)
    
    hue_differences = []
    saturation_scales = []
    value_scales = []
    
    for color in ["blue", "green", "red"]:
        
        expected_hsv_range = expected_hsv_ranges[color].astype(float)
        observed_hsv = observed_hsv_dictionary[color].astype(float)

        hue_differences.append(calculate_hue_difference(float(expected_hsv_range[0]), float(observed_hsv[0])))  
        saturation_scales.append(expected_hsv_range[1] / max(1.0, observed_hsv[1]))
        value_scales.append(expected_hsv_range[2] / max(1.0, observed_hsv[2]))
    
    average_hue_offset = np.mean(hue_differences)

    saturation_scale = np.median(saturation_scales)
    value_scale = np.median(value_scales)
    
    print("\n=== Average HSV offsets applied ===")
    print(f"Average H offset: {average_hue_offset:.2f}")
    print(f"S scale: {saturation_scale:.2f}")
    print(f"V scale: {value_scale:.2f}\n")
    
    corrected_ranges = {}

    for color, (lower, upper) in original_hsv_ranges.items():

        lower = np.array(lower, dtype=float)
        upper = np.array(upper, dtype=float)

        lower_h = (lower[0] + average_hue_offset) % 180
        upper_h = (upper[0] + average_hue_offset) % 180

        lower_sv = np.clip(lower[1:] * np.array([saturation_scale, value_scale]), 0, 255)
        upper_sv = np.clip(upper[1:] * np.array([saturation_scale, value_scale]), 0, 255)

        lower_corrected = np.array([lower_h, lower_sv[0], lower_sv[1]])
        upper_corrected = np.array([upper_h, upper_sv[0], upper_sv[1]])

        lower_corrected = np.clip(lower_corrected, [0,0,0], [179,255,255]).astype(int)
        upper_corrected = np.clip(upper_corrected, [0,0,0], [179,255,255]).astype(int)

        corrected_ranges[color] = (lower_corrected, upper_corrected)
        
    return original_hsv_ranges
