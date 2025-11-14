
# --- Imports ---

import cv2
import numpy as np

from utilities.global_definitions import (
    reference_match_threshold,
    start_frame_color, start_frame_detection_tolerance,
    end_frame_color, end_frame_detection_tolerance
)

# --- Functions ---

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

def detect_start_frame(frame):

    """
    Detects if the average frame color is close to the start frame color.

    Arguments:
        "frame" (np.ndarray): A NumPy array representing the frame pixels.

    Returns:
        bool: True if the average frame color is close to the start frame color, False otherwise.
    
    """

    average_frame_color = frame.mean(axis = (0, 1)) # Calculates the average color of the frame

    distance = np.linalg.norm(average_frame_color - np.array(start_frame_color)) # Calculates the Euclidean distance between the average frame color and the start frame color

    return distance < start_frame_detection_tolerance # Returns True if the distance is less than the tolerance, False otherwise

def detect_end_frame(frame):

    """
    Detects if the average frame color is close to the end frame color.

    Arguments:
        "frame" (np.ndarray): A NumPy array representing the frame pixels.

    Returns:
        bool: True if the average frame color is close to the end frame color, False otherwise.
    
    """

    average_frame_color = frame.mean(axis = (0, 1)) # Calculates the average color of the frame

    distance = np.linalg.norm(average_frame_color - np.array(end_frame_color)) # Calculates the Euclidean distance between the average frame color and the end frame color

    return distance < end_frame_detection_tolerance # Returns True if the distance is less than the tolerance, False otherwise