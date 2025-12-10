
# --- Imports ---

# Library modules

import time
import cv2
import numpy as np
from numba import njit, prange

# Non-library modules

from utilities.global_definitions import (
    number_of_rows, number_of_columns,
    red_lower_hcv_limit_1, red_upper_hcv_limit_1,
    red_lower_hcv_limit_2, red_upper_hcv_limit_2,
    white_lower_hcv_limit, white_upper_hcv_limit,
    black_lower_hcv_limit, black_upper_hcv_limit,
    green_lower_hcv_limit, green_upper_hcv_limit,
    blue_lower_hcv_limit, blue_upper_hcv_limit,
    end_bit_steps, dominant_color_steps, idx_to_2bit, 
    idx_to_1bit, idx_to_3bit, bits_per_cell
)

# --- Functions ---

class BitColorTracker:

    """
    
    """

    def __init__(self):
        
        """
        
        """

        self.time_in = 0
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns
        self.hcv_frames = []
        self.LUT = None
        self.color_names = None

    def add_frame(self, hcv_roi):
        
        """
        
        """

        self.hcv_frames.append(hcv_roi)

        if not hasattr(BitColorTracker.add_frame, "time"):
            self.time_in = time.time()
            BitColorTracker.add_frame.time = ("chingchong")

        if not hasattr(BitColorTracker.add_frame, "walla") and (time.time() - self.time_in > 0.15) and self.time_in != 0:
            
            H = hcv_roi[:,:,0]; S = hcv_roi[:,:,1]; V = hcv_roi[:,:,2]

            print("hcv roi shape:", hcv_roi.shape, "dtype:", hcv_roi.dtype)
            print("H min/max:", int(H.min()), int(H.max()))
            print("S min/max:", int(S.min()), int(S.max()))
            print("V min/max:", int(V.min()), int(V.max()))

            BitColorTracker.add_frame.walla = ("bingbing")

    def _pad_frames(self, frames):

        
        """
        
        """

        _, H, W, _ = frames.shape

        cell_h = int(np.ceil(H / self.number_of_rows))
        cell_w = int(np.ceil(W / self.number_of_columns))

        padded_H = cell_h * self.number_of_rows
        padded_W = cell_w * self.number_of_columns

        pad_bottom = padded_H - H
        pad_right = padded_W - W

        padded_frames = np.pad(frames, ((0, 0), (0, pad_bottom), (0, pad_right), (0, 0)), mode = "edge")

        return padded_frames, cell_h, cell_w
  
    def end_bit(self):

        """
        
        """

        if len(self.hcv_frames) == 0:
            return None

        hcv_frames = np.asarray(self.hcv_frames)
        self.hcv_frames = []

        padded_frames, self.cell_h, self.cell_w = self._pad_frames(hcv_frames)

        N, _, _, _ = padded_frames.shape

        cells = padded_frames.reshape(N, self.number_of_rows, self.cell_h, self.number_of_columns, self.cell_w, 3)

        patch_h = max(self.cell_h // 2, 1)
        patch_w = max(self.cell_w // 2, 1)

        h0 = (self.cell_h - patch_h) // 2
        h1 = h0 + patch_h
        w0 = (self.cell_w - patch_w) // 2
        w1 = w0 + patch_w

        sampled_cells = cells[:, :, h0:h1, :, w0:w1, :]

        if patch_h > end_bit_steps and patch_w > end_bit_steps:
            sampled_cells = sampled_cells[:, :, ::end_bit_steps, :, ::end_bit_steps, :]

        Hc = sampled_cells[..., 0].astype(np.uint16)
        Cc = sampled_cells[..., 1].astype(np.uint16)
        Vc = sampled_cells[..., 2].astype(np.uint16)

        classes = self.LUT[Hc, Cc, Vc]

        merged = classes

        number_of_classes = int(self.LUT.max()) + 1
        bitgrid = bitgrid_majority_calculator(merged, number_of_classes)

        if bits_per_cell == 1:
            idx_to_bit = idx_to_1bit
        elif bits_per_cell == 2:
            idx_to_bit = idx_to_2bit
        elif bits_per_cell == 3:
            idx_to_bit = idx_to_3bit
        else:
            raise NotImplementedError("Mapping table for this bits_per_cell not defined")

        bitgrid_2bit = np.vectorize(lambda idx: idx_to_bit.get(idx, 0b00))(bitgrid)
        
        return bitgrid_2bit

    def bitgrid_list_to_bitstream(bitgrids, bits_per_cell):

        bitstream = ""

        for grid in bitgrids:
            for val in grid.ravel():
                bitstream += format(val, f"0{bits_per_cell}b")  # Convert to bits_per_cell-bit binary string

        return bitstream
    
    def reset(self):
        
        """
        
        """

        self.hcv_frames = []

    def colors(self, LUT, color_names):
        
        """
        
        """

        self.LUT = LUT
        self.color_names = color_names

@njit(parallel = True)
def bitgrid_majority_calculator(patch_class_array, number_of_classes):

    """
    Aggregates patches of class labels for each cell in a grid and assigns the cell its most frequently occurring label.

    Arguments:
        "patch_class_array": A 5-dimensional array (tensor).
        "number_of_classes": The total number of distinct class labels that "patch_class_array" can contain.
    
    Returns:
        "out": A grid of majority labels.

    """

    number_of_frames, number_of_grid_rows, patch_heights, number_of_grid_columns, patch_widths = patch_class_array.shape # Extracts the five dimensions of "patch_class_array"

    out = np.empty((number_of_grid_rows, number_of_grid_columns), dtype = np.int32) # Creates an empty integer array to store the majority class for each grid coordinate

    for row in prange(number_of_grid_rows): # For each row:

        for column in range(number_of_grid_columns): # For each column:

            counts = np.zeros(number_of_classes, dtype = np.int32) # Create a histogram array to accumulate how often each class appears in the cell region

            for sample in range(number_of_frames): # For each frame:

                for patch_height in range(patch_heights): # Loop over the patch height:

                    for patch_width in range(patch_widths): # Loop over the patch width:

                        class_id = patch_class_array[sample, row, patch_height, column, patch_width] # Extract the class ID at these coordinates

                        if 0 <= class_id < number_of_classes: # If the class ID number is within the given class ID range
                            counts[class_id] += 1 # Increase the counter for that class ID

            majority_class_id = 0 # Initialize the majority class id as 0
            majority_class_count = counts[majority_class_id] # Initializes the majority class value

            for class_id_index in range(1, number_of_classes): # Argmax to find the class with the highest frequency

                if counts[class_id_index] > majority_class_count:
                    majority_class_id = class_id_index
                    majority_class_count = counts[class_id_index]

            out[row, column] = majority_class_id # Assign the majority class to the output grid

    return out

tracker = BitColorTracker()

def bgr_to_hcv(bgr):
    B = bgr[..., 0].astype(np.int16)
    G = bgr[..., 1].astype(np.int16)
    R = bgr[..., 2].astype(np.int16)

    maxc = np.maximum.reduce([R, G, B])
    minc = np.minimum.reduce([R, G, B])
    C = maxc - minc
    V = maxc

    h = np.zeros_like(R)

    mask = C != 0
    idx = (maxc == R) & mask
    h[idx] = (60 * (G[idx] - B[idx]) // C[idx]) % 360

    idx = (maxc == G) & mask
    h[idx] = (60 * (B[idx] - R[idx]) // C[idx] + 120) % 360

    idx = (maxc == B) & mask
    h[idx] = (60 * (R[idx] - G[idx]) // C[idx] + 240) % 360

    # Convert to OpenCV-like hue scaling
    H = (h // 2).astype(np.uint8)
    C = C.astype(np.uint8)
    V = V.astype(np.uint8)

    # STACK INTO (H, W, 3) IMAGE
    return np.stack((H, C, V), axis=-1)



def build_color_LUT(corrected_ranges):
    """
    Build a 180 x 256 x 256 LUT mapping HCV -> class index.
    H = 0..179
    C = 0..255  (chroma)
    V = 0..255
    """

    color_names = list(corrected_ranges.keys())
    LUT = np.zeros((180, 256, 256), dtype=np.uint8)

    # Build lookup grids
    H = np.broadcast_to(np.arange(180, dtype=np.uint16)[:, None, None], (180, 256, 256))
    C = np.broadcast_to(np.arange(256, dtype=np.uint16)[None, :, None], (180, 256, 256))
    V = np.broadcast_to(np.arange(256, dtype=np.uint16)[None, None, :], (180, 256, 256))

    for idx, (_, (lower, upper)) in enumerate(corrected_ranges.items()):
        lh, lc, lv = lower
        uh, uc, uv = upper

        # Hue wrapping logic remains the same
        if lh <= uh:
            mask = (
                (H >= lh) & (H <= uh) &
                (C >= lc) & (C <= uc) &
                (V >= lv) & (V <= uv)
            )
        else:
            mask = (
                ((H >= lh) | (H <= uh)) &
                (C >= lc) & (C <= uc) &
                (V >= lv) & (V <= uv)
            )

        LUT[mask] = idx

    return LUT, color_names


def dominant_color_hcv(hcv):

    LUT = tracker.LUT
    names = tracker.color_names
    
    ph, pw, _ = hcv.shape
    
    if ph > dominant_color_steps and pw > dominant_color_steps:
        hcv = hcv [::dominant_color_steps, ::dominant_color_steps, :]

    H = hcv[:, :, 0]
    C = hcv[:, :, 1]
    V = hcv[:, :, 2]

    classes = LUT[H, C, V]

    hist = np.bincount(classes.ravel(), minlength = len(names))

    names = names[int(hist.argmax())]

    if names == "red1" or names == "red2":
        return "red"
    
    else:
        return names


def color_offset_calculation(roi):

    """
    Calculates color offsets based on a calibration ROI containing red, green, and blue stripes.

    Arguments:
        "roi" (numpy.ndarray): The region of interest image in BGR format.
    
    Returns:
        "corrected_hcv_ranges" (dict): A dictionary containing the corrected hcv ranges for various colors.

    """

    expected_hcv_ranges = {
    "blue": np.array([120, 255, 255]),
    "green": np.array([60, 255, 255]),
     "red": np.array([0, 255, 255])
    }
    """
    original_hcv_ranges = {
        "red1":  (red_lower_hcv_limit_1, red_upper_hcv_limit_1),
        "red2":  (red_lower_hcv_limit_2, red_upper_hcv_limit_2),
        "white": (white_lower_hcv_limit, white_upper_hcv_limit),
        "black": (black_lower_hcv_limit, black_upper_hcv_limit),
        "green": (green_lower_hcv_limit, green_upper_hcv_limit),
        "blue":  (blue_lower_hcv_limit , blue_upper_hcv_limit )
    }
    """
    original_hcv_ranges = { "white": ([0, 0, 200], [179, 40, 255]),
                            "black": ([0, 0, 0], [179, 255, 50]), 
                            "red1": ([0, 40, 60], [10, 255, 255]),
                            "red2": ([160, 40, 60], [179, 255, 255]),
                            "green": ([40, 40, 60], [80, 255, 255]), 
                            "blue": ([100, 40, 60], [140, 255, 255]),
                            "yellow":([20,40,60],[40,255,255]),
                            "cyan": ([80,40, 60],[100,255,255]),
                            "magenta":([140,40,60],[160,255,255]),
                            "gray": ([0, 0, 80], [179, 50, 200]), 
                            "orange": ([10, 40, 60],[20, 255, 255])}


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
        
    roi_hcv = bgr_to_hcv(roi) # Converts the ROI to hcv format
    
    stripe_width = roi.shape[1] // 3 # Width of each color stripe
    patch_width = int(stripe_width * 0.5) # Width of the patch to sample within each stripe
    start_offset = (stripe_width - patch_width) // 2 # Offset to center the patch within the stripe

    observed_hcv_dictionary = {}
    
    for stripe_index, color in enumerate(["blue", "green", "red"]):

        x_start = stripe_index * stripe_width + start_offset
        x_end = x_start + patch_width

        roi_stripe = roi_hcv[:, x_start:x_end]

        observed_hcv_dictionary[color] = np.median(roi_stripe.reshape(-1,3), axis = 0)
    
    hue_differences = []
    chroma_scales = []
    value_scales = []
    
    for color in ["blue", "green", "red"]:
        
        expected_hcv_range = expected_hcv_ranges[color].astype(float)
        observed_hcv = observed_hcv_dictionary[color].astype(float)

        hue_differences.append(calculate_hue_difference(float(expected_hcv_range[0]), float(observed_hcv[0])))  
        chroma_scales.append(expected_hcv_range[1] / max(1.0, observed_hcv[1]))
        value_scales.append(expected_hcv_range[2] / max(1.0, observed_hcv[2]))
    
    average_hue_offset = np.mean(hue_differences)

    chroma_scale = np.median(chroma_scales)
    value_scale = np.median(value_scales)
    
    print("\n=== Average hcv offsets applied ===")
    print(f"Average H offset: {average_hue_offset:.2f}")
    print(f"C scale: {chroma_scale:.2f}")
    print(f"V scale: {value_scale:.2f}\n")
    
    corrected_ranges = {}

    for color, (lower, upper) in original_hcv_ranges.items():

        lower = np.array(lower, dtype=float)
        upper = np.array(upper, dtype=float)

        lower_h = (lower[0] + average_hue_offset) % 180
        upper_h = (upper[0] + average_hue_offset) % 180

        lower_sv = np.clip(lower[1:] * np.array([chroma_scale, value_scale]), 0, 255)
        upper_sv = np.clip(upper[1:] * np.array([chroma_scale, value_scale]), 0, 255)

        lower_corrected = np.array([lower_h, lower_sv[0], lower_sv[1]])
        upper_corrected = np.array([upper_h, upper_sv[0], upper_sv[1]])

        lower_corrected = np.clip(lower_corrected, [0,0,0], [179,255,255]).astype(int)
        upper_corrected = np.clip(upper_corrected, [0,0,0], [179,255,255]).astype(int)

        corrected_ranges[color] = (lower_corrected, upper_corrected)
        
    return corrected_ranges