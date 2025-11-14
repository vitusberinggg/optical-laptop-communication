
# --- Imports ---

import cv2
import numpy as np

from global_definitions import (
    laptop_webcam_pixel_height, laptop_webcam_pixel_width,
    sender_output_height, sender_output_width
)

# --- Functions ---

def create_mask(homography_matrix):

    """
    Creates a binary mask using a homography matrix.

    Arguments:
        "homography_matrix"

    Returns:
        "mask"

    """

    sender_mask = np.full((sender_output_height, sender_output_width), 255, np.uint8)

    inverse_homography_matrix = np.linalg.inv(homography_matrix)

    webcam_mask = cv2.warpPerspective(
        sender_mask,
        inverse_homography_matrix,
        (laptop_webcam_pixel_width, laptop_webcam_pixel_height),
        flags = cv2.INTER_NEAREST
    )

    return webcam_mask

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