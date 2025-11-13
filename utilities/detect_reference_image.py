
# --- Imports ---

import cv2

from utilities.global_definitions import reference_match_threshold

# --- Main function ---

def detect_reference_image(potential_sync_frame, reference_image):

    """
    Detects if the reference image is present in the potential sync frame.

    Arguments:
        "potential_sync_frame" (np.ndarray): A NumPy array representing the potential sync frame pixels.
        "reference_image" (np.ndarray): A NumPy array representing the reference image pixels.

    Returns:
        tuple: A tuple containing a boolean indicating if the reference image is detected and the maximum correlation value.

    """

    resized_frame = cv2.resize(potential_sync_frame, (reference_image.shape[1], reference_image.shape[0])) # Resize the potential sync frame to match the reference image size

    correlation = cv2.matchTemplate(resized_frame, reference_image, cv2.TM_CCOEFF_NORMED) # Perform template matching to find the reference image in the potential sync frame

    max_correlation = cv2.minMaxLoc(correlation)[1] # Get the maximum correlation value

    return max_correlation > reference_match_threshold, max_correlation # Return whether the reference image is detected and the maximum correlation value