
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

def create_aruco_marker_frame(position="right"):
    """
    Creates a frame with a single vertical ArUco marker, using global marker size and margin.

    The marker:
      - is a square of side length aruco_marker_size
      - is centered vertically with 15px margin top and bottom
      - is 15px from the left or right edge depending on `position`
    """
    frame = create_color_frame([0, 255, 0])  # Green background

    marker_size = aruco_marker_size         # side length of the square marker
    margin = aruco_marker_margin

    # Sanity check: make sure it fits vertically
    if marker_size + 2 * margin > sender_output_height:
        raise ValueError(
            f"Marker (size={marker_size}) + margins (2*{margin}) "
            f"doesn't fit in height={sender_output_height}"
        )

    # Vertical placement: margin at top and bottom
    y_coordinate = margin  # top of the marker
    # bottom will be y_coordinate + marker_size = sender_output_height - margin

    # Horizontal position using margin
    if position == "right":
        x_coordinate = sender_output_width - margin - marker_size
    elif position == "left":
        x_coordinate = margin
    else:
        raise ValueError("position must be 'left' or 'right'")

    # Use first marker ID (or change later if you like)
    aruco_marker_id = aruco_marker_ids[0]

    # Generate marker (square)
    marker = cv2.aruco.generateImageMarker(
        aruco_marker_dictionary,
        aruco_marker_id,
        marker_size
    )
    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    # Paste marker onto frame
    frame[y_coordinate:y_coordinate + marker_size,
          x_coordinate:x_coordinate + marker_size] = marker_bgr

    return frame


def create_color_reference_frame():
    
    """
    Creates a reference frame with all key colors for the receiver to calibrate.

    Arguments: 
        None
    
    Returns:
        ref_frame (np.ndarray): The reference frame (BGR).
    """
    # Create blank frame
    color_reference_frame = np.zeros((sender_output_height, sender_output_width, 3), dtype=np.uint8)

    # Divide frame into equal vertical stripes for each color
    colors = [sync_frame_color, start_frame_color, end_frame_color]
    num_colors = len(colors)
    stripe_width = sender_output_width // num_colors

    for i, color in enumerate(colors):
        x_start = i * stripe_width
        x_end = (i + 1) * stripe_width if i != num_colors - 1 else sender_output_width
        ref_frame[:, x_start:x_end] = color

    return color_reference_frame
