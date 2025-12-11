
# --- Imports ---

# Library imports

import numpy as np
import cv2

# Non-library imports

from utilities.global_definitions import(
    sender_output_height, sender_output_width,
    blue_bgr, green_bgr, red_bgr, gray_bgr,
    number_of_rows, number_of_columns,
    bit_cell_width, bit_cell_height,
    aruco_marker_margin, large_aruco_marker_side_length,
    aruco_marker_ids, aruco_marker_dictionary
)

# --- Functions ---

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

def render_multicolor_frame(bitgrid, color_map):

    """
    Renders a 2D bitgrid as an image by mapping each cell value to a color and drawing rectangles.

    Arguments:
        "bitgrid" (np.array): A 2D numpy array where each element represents a cell value to be rendered.
        "color_map": A map holding information about what cell values represent what colors.

    Returns:
        "image" (np.array): A numpy array representing the rendered image in BGR format.

    """

    image = np.zeros((sender_output_height, sender_output_width, 3), dtype = np.uint8)

    for row in range(number_of_rows):

        for column in range(number_of_columns):

            cell_value = int(bitgrid[row, column])
            color = color_map[cell_value]  # Get the BGR color

            # Cell coordinates
            x0 = column * bit_cell_width
            x1 = x0 + bit_cell_width
            y0 = row * bit_cell_height
            y1 = y0 + bit_cell_height

            cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), color, thickness = -1)

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