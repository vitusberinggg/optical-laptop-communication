
# --- Imports ---

import numpy as np
import cv2

from utilities.global_definitions import (
    sender_output_height, sender_output_width,
    number_of_columns, number_of_rows, 
    bit_cell_height, bit_cell_width,
    reference_image_seed,
    aruco_marker_dictionary, aruco_marker_size, aruco_marker_margin, aruco_marker_ids
)

# --- Functions ---

def render_frame(bitgrid):

    """
    Renders a frame from a NumPy array.

    Arguments:
        "bitgrid" (np.ndarray): A 2D NumPy array of bits.

    Returns:
        "image" (np.ndarray): A NumPy array representing the image frame pixels.

    """

    image = np.zeros((sender_output_height, sender_output_width, 3), dtype = np.uint8)

    for row in range(number_of_rows): # For each row:

        for column in range(number_of_columns): # For each column:

            bit = int(bitgrid[row, column]) # Get the bit at the current position

            if bit == 1: # If the bit is 1:
                color = (255, 255, 255) # Set the color to white
            
            else: # Else (if the bit is 0):
                color = (0, 0, 0) # Set the color to black

            start_x_coordinate = column * bit_cell_width
            end_x_coordinate = start_x_coordinate + bit_cell_width

            start_y_coordinate = row * bit_cell_height
            end_y_coordinate = start_y_coordinate + bit_cell_height

            cv2.rectangle(image, (start_x_coordinate, start_y_coordinate), (end_x_coordinate - 1, end_y_coordinate - 1), color, thickness = -1) # Draw the rectangle on the image

    return image

def generate_reference_image():

    """
    Generates a reference image by putting the image seed into a bit generator.

    Arguments:
        None

    Returns:
        "reference_image": The reference image.

    """

    bit_generator = np.random.RandomState(reference_image_seed)
    reference_image = bit_generator.randint(0, 256, (sender_output_height, sender_output_width), dtype = np.uint8)

    return reference_image

def create_color_frame(color):

    """
    Creates a solid color frame.

    Arguments:
        "color" (tuple): A tuple representing the BGR color.

    Returns:
        "frame" (np.ndarray): A NumPy array representing the solid color frame pixels.

    """

    return np.full((sender_output_height, sender_output_width, 3), color, dtype = np.uint8)

def create_aruco_marker_frame():

    """
    Creates a solid color frame with ArUco markers in each corner.

    Arguments:
        None

    Returns:
        "frame": The created frame.

    """

    frame = create_color_frame([0, 255, 0])

    aruco_marker_positions = [
        (aruco_marker_margin, aruco_marker_margin), # Top-left marker
        (sender_output_width - aruco_marker_margin - aruco_marker_size, aruco_marker_margin), # Top-right marker
        (aruco_marker_margin, sender_output_height - aruco_marker_margin - aruco_marker_size), # Bottom-left marker
        (sender_output_width - aruco_marker_margin - aruco_marker_size, sender_output_height - aruco_marker_margin - aruco_marker_size) # Bottom-right marker
    ]

    for (x_coordinate, y_coordinate), aruco_marker_id in zip(aruco_marker_positions, aruco_marker_ids):
        marker = cv2.aruco.generateImageMarker(aruco_marker_dictionary, aruco_marker_id, aruco_marker_size)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        frame[y_coordinate:y_coordinate + aruco_marker_size, x_coordinate:x_coordinate + aruco_marker_size] = marker_bgr

    return frame