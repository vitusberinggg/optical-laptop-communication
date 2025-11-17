
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

def create_mask_2(homography_matrix):
    """
    Creates a binary mask. 
    Calculates the inverse of the homography to map Sender dimensions -> Camera dimensions.
    """
    h, w = sender_output_height, sender_output_width
    
    pts_src = np.array([
        [0, 0], 
        [w, 0], 
        [w, h], 
        [0, h]
    ], dtype=np.float32)
    
    pts_src = np.array([pts_src])

    try:
        inv_homography = np.linalg.inv(homography_matrix)
    except np.linalg.LinAlgError:
        print("[ERROR] Homography matrix is singular and cannot be inverted.")
        return np.zeros((laptop_webcam_pixel_height, laptop_webcam_pixel_width), dtype=np.uint8)

    pts_dst = cv2.perspectiveTransform(pts_src, inv_homography)
    
    # Reshape from (1, 4, 2) to (4, 1, 2) - THIS IS THE KEY!
    pts_dst = pts_dst.reshape(-1, 1, 2)
    
    print(f"[DEBUG] All transformed corners:\n{pts_dst}")
    print(f"[DEBUG] Shape: {pts_dst.shape}")

    x_coord = pts_dst[0][0][0]
    y_coord = pts_dst[0][0][1]
    
    print(f"[DEBUG] Corrected Mask Coordinates (Top Left): x={x_coord:.2f}, y={y_coord:.2f}")
    
    if not (0 <= x_coord <= laptop_webcam_pixel_width) or not (0 <= y_coord <= laptop_webcam_pixel_height):
        print(f"[WARNING] The mask is starting OUTSIDE the camera frame. Adjust your ArUco detection.")

    webcam_mask = np.zeros((laptop_webcam_pixel_height, laptop_webcam_pixel_width), dtype=np.uint8)

    cv2.fillConvexPoly(webcam_mask, np.int32(pts_dst), 255)

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

    captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

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