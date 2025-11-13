
# --- Imports ---

import numpy as np

from utilities.global_definitions import end_frame_color, end_frame_detection_tolerance

# --- Main function ---

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
