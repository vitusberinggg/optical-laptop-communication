
# --- Imports ---

import numpy as np

from utilities.global_definitions import start_frame_color, start_frame_detection_tolerance

# --- Main function ---

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
