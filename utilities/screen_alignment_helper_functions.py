
# --- Imports ---

import cv2
import numpy as np

from global_definitions import mask_frame_hsv_lower_limit, mask_frame_hsv_upper_limit, total_pixel_count, sender_screen_size_threshold

# --- Functions ---

def create_mask(frame):

    """
    Creates a binary mask by detecting a lime green screen in a frame.

    Arguments:
        "frame"

    Returns:
        "mask"

    """

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Converts the frame to HSV (Hue, Saturation, Value)

    lower_green = np.array(mask_frame_hsv_lower_limit)
    upper_green = np.array(mask_frame_hsv_upper_limit)

    mask = cv2.inRange(hsv_frame, lower_green, upper_green) # Creates a mask by setting pixels within the green range to 255, and pixels outside to 0
    
    green_pixel_count = np.count_nonzero(mask) # Counts the amount of green pixels

    if green_pixel_count > (total_pixel_count * sender_screen_size_threshold):
        print(f"[INFO] Green screen covering {green_pixel_count // total_pixel_count} of the frame found.")
        return mask
    
    else:
        print("No green screen found.")
        return None


def compute_ecc_transform(reference_image_float, captured_frame_float, initial_warp_matrix, warp_mode, criteria):
    
    """

    """
    
    # Get the target size from the reference image.
    target_height, target_width = reference_image_float.shape

    # Resize the captured frame to match the reference image's size.
    resized_captured_frame = cv2.resize(
        captured_frame_float, 
        (target_width, target_height), 
        interpolation=cv2.INTER_LINEAR
    )
    
    # --- This is the mask fix we discussed ---
    # We assume the screen is brighter (e.g., > 80) than the background.
    # This mask tells ECC to *only* look at the bright parts of the captured image.
    # You MUST tune this threshold (80) for your lighting!
    brightness_threshold = 80
    _ignored, input_mask = cv2.threshold(
        resized_captured_frame.astype(np.uint8),  # threshold needs uint8
        brightness_threshold, 
        255, 
        cv2.THRESH_BINARY
    )
    # --- End of mask fix ---

    try:
        # Run the ECC algorithm.
        print("[INFO] Computing ECC warp... (this may take a moment)")
        correlation_coefficient, warp_matrix = cv2.findTransformECC(
            reference_image_float,   # The template (what we want)
            resized_captured_frame,  # The input (what we see)
            initial_warp_matrix,     # Our starting guess (identity matrix)
            warp_mode,               # Type of motion (AFFINE)
            criteria,                # When to stop
            input_mask,              # **THE MASK**: Ignore background
            5                        # Add Gaussian blur to help the algorithm
        )
        print(f"[INFO] ECC warp computed (correlation={correlation_coefficient:.6f}).")
        
        # Return success and the new matrix.
        return True, warp_matrix
        
    except cv2.error as e:
        # If ECC fails (e.g., images are too different), print a warning.
        print(f"[WARN] findTransformECC failed: {e}. Will retry.")
        
        # Return failure and the original matrix.
        return False, initial_warp_matrix