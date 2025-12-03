
# --- Imports ---

import numpy as np
import cv2

from utilities.global_definitions import (
    red_bgr, green_bgr, blue_bgr, gray_bgr,
    sender_output_height, sender_output_width,
    number_of_columns, number_of_rows, 
    bit_cell_height, bit_cell_width,
    aruco_marker_dictionary, small_aruco_marker_side_length, large_aruco_marker_side_length, aruco_marker_margin, aruco_marker_ids
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
        (sender_output_width - aruco_marker_margin - small_aruco_marker_side_length, aruco_marker_margin), # Top-right marker
        (aruco_marker_margin, sender_output_height - aruco_marker_margin - small_aruco_marker_side_length), # Bottom-left marker
        (sender_output_width - aruco_marker_margin - small_aruco_marker_side_length, sender_output_height - aruco_marker_margin - small_aruco_marker_side_length) # Bottom-right marker
    ]

    for (x_coordinate, y_coordinate), aruco_marker_id in zip(aruco_marker_positions, aruco_marker_ids):
        marker = cv2.aruco.generateImageMarker(aruco_marker_dictionary, aruco_marker_id, small_aruco_marker_side_length)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        frame[y_coordinate:y_coordinate + small_aruco_marker_side_length, x_coordinate:x_coordinate + small_aruco_marker_side_length] = marker_bgr

    return frame

def create_large_aruco_marker_frame(position = "right"):
    
    """
    Creates a gray frame with a single large ArUco marker on the left or right.

    Arguments:
        position (str): "right" or "left" side for the marker (default "right").

    Returns:
        np.ndarray: The frame with the large ArUco marker.
        
    """

    frame = create_color_frame(gray_bgr)

    y_coordinate = aruco_marker_margin

    if position == "right":
        x_coordinate = sender_output_width - aruco_marker_margin - large_aruco_marker_side_length

    elif position == "left":
        x_coordinate = aruco_marker_margin

    else:
        raise ValueError("position must be 'left' or 'right'")
    
    if position == "right":
        aruco_marker_id = aruco_marker_ids[0]

    else:
       aruco_marker_id = aruco_marker_ids[1]

    marker = cv2.aruco.generateImageMarker(aruco_marker_dictionary, aruco_marker_id, large_aruco_marker_side_length)
    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    frame[y_coordinate:y_coordinate + large_aruco_marker_side_length, x_coordinate:x_coordinate + large_aruco_marker_side_length] = marker_bgr

    return frame

def create_color_reference_frame():
    
    """
    Creates a reference frame with all key colors for the receiver to calibrate.

    Arguments: 
        None
    
    Returns:
        color_reference_frame (np.ndarray): The reference frame (BGR).

    """

    color_reference_frame = np.zeros((sender_output_height, sender_output_width, 3), dtype = np.uint8) # Creates a blank frame

    colors = [blue_bgr, green_bgr, red_bgr]

    stripe_width = sender_output_width // len(colors) # Divides the frame into equal vertical stripes for each color

    for stripe_index, color in enumerate(colors):

        x_start = stripe_index * stripe_width

        if stripe_index != len(colors) - 1: # If the stripe index isn't the last one:
            x_end = (stripe_index + 1) * stripe_width
        
        else: # Else (if it's the last one):
            x_end = sender_output_width

        color_reference_frame[:, x_start:x_end] = color # Fill the entire stripe with the current color

    return color_reference_frame
