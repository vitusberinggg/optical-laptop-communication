
# --- Imports ---

import cv2
import numpy as np

from utilities.global_definitions import (
    laptop_webcam_pixel_height, laptop_webcam_pixel_width,
    sender_output_height, sender_output_width,
    ecc_allignment_criteria
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

def compute_ecc_transform(reference_image, captured_image):
    
    """
    Alligns the captured reference image to the reference image using ECC (Enhanced Correlation Coefficient).

    Arguments:
        "reference_image": The reference image.
        "captured_image": The image to allign.

    Returns:
        "ecc_warp_matrix": The warp matrix.

    """

    ecc_warp_matrix = np.eye(2, 3, dtype = np.float32) # Initial warp matrix guess

    try:
        cc, ecc_warp_matrix = cv2.findTransformECC(
            reference_image,
            captured_image,
            ecc_warp_matrix,
            cv2.MOTION_AFFINE,
            ecc_allignment_criteria
        )        

    except cv2.error:

        print("[WARNING] ECC allignment failed.")
        return None
    
    return ecc_warp_matrix