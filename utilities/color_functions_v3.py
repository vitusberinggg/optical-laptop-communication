
# --- Imports ---

import cv2
import numpy as np
from collections import Counter
from numba import njit, prange
from utilities.global_definitions import number_of_rows as rows, number_of_columns as cols

# --- Functions ---

class BitColorTracker:
    def __init__(self):
        self.rows = rows
        self.cols = cols
        self.hsv_frames = []

    def add_frame(self, hsv_roi):
        self.hsv_frames.append(hsv_roi)

    def _pad_frames(self, frames):
        # H = frame height, W = frame width
        N, H, W, C = frames.shape

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
        # checks if hsv_frames has any elements in it, if so return None to prevent an exception
        if len(self.hsv_frames) == 0:
            return None

        # hsv_frames = all hsv in an array
        # if the given hsv is not divisible by the number of rows/cols in height/width it will padd the hsv
        # hsv = frame but is read in hsv and not bgr
        hsv_frames = np.asarray(self.hsv_frames)
        padded_frames, self.cell_h, self.cell_w = self._pad_frames(hsv_frames)
        self.hsv_frames = []

        N, H, W, _ = padded_frames.shape
        # N = number frames, H = height of padded frame, W = width of the padded frame

        # split padded frame into grid cells
        cells = padded_frames.reshape(
            N, self.rows, self.cell_h,
            self.cols, self.cell_w, 3
        )

        # size of the smaller sample inside each cell
        ds_h = max(self.cell_h // 4, 1)
        ds_w = max(self.cell_w // 4, 1)

        # making the smaller sample with the given sizes
        sampled_cells = cells[:, :, ::ds_h, :, ::ds_w, :]

        # HSV → class IDs via LUT
        Hc = sampled_cells[..., 0].astype(np.uint16)
        Sc = sampled_cells[..., 1].astype(np.uint16)
        Vc = sampled_cells[..., 2].astype(np.uint16)

        classes = self.LUT[Hc, Sc, Vc]    # shape: (N, rows, ds_h, cols, ds_w)

        # collapse all samples inside each cell to 1 dimension:
        # for each frame n, row r, col c, collect all sample values
        N, R, Sh, C, Sw = classes.shape
        num_samples = Sh * Sw
        # num_samples basically means all the pixels inside the sample in each cell

        merged = classes.reshape(N, R, C, num_samples)
        # merged.shape = (frames, rows, cols, samples_per_cell)

        # makes a variable that has the value of the number of different colors in the Lookup Table
        num_classes = int(self.LUT.max()) + 1

        bitgrid = bitgrid_majority_calc(merged, num_classes)  # bitgrid(rows, cols)

        # checks if its white or not
        white_idx = 1
        bitgrid_str = np.where(bitgrid == white_idx, "1", "0")

        return bitgrid_str

    def reset(self):
        self.hsv_frames = []

    def colors(self, LUT, color_names):
        self.LUT = LUT
        self.color_names = color_names

# --- Numba helper for majority vote ---
@njit(parallel=True)
def bitgrid_majority_calc(merged, num_classes):
    # merged shape: (N_frames, R_rows, C_cols, S_samples)
    N, R, C, S = merged.shape

    out = np.empty((R, C), dtype=np.int32)

    # parallell looping for speed
    for r in prange(R):
        for c in range(C):

            # histogram will help us calc the dominant color
            # histogram for this cell
            counts = np.zeros(num_classes, dtype=np.int32)

            # accumulate across all frames and samples inside the cell
            for n in range(N):
                for s in range(S):
                    cls = merged[n, r, c, s]
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

@njit
def dominant_color_numba(classes, num_colors):
    hist = np.zeros(num_colors, dtype=np.int32)
    for i in range(classes.size):
        hist[classes.ravel()[i]] += 1
    return np.argmax(hist)


def dominant_color(hsv, sample_step=4):
    """
    Computes the dominant color in an HSV frame by sampling every `sample_step` pixel.
    """
    LUT = tracker.LUT
    names = tracker.color_names

    # Sample every `sample_step` pixel in height and width
    H = hsv[::sample_step, ::sample_step, 0]
    S = hsv[::sample_step, ::sample_step, 1]
    V = hsv[::sample_step, ::sample_step, 2]

    # LUT lookup for sampled pixels
    classes = LUT[H, S, V]

    # Majority vote
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
    saturation_differences = []
    value_differences = []
    
    for color in ["blue", "green", "red"]:
        
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
        
    return original_hsv_ranges