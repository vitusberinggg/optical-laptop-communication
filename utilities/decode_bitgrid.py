
# --- Imports ---

import numpy as np

from utilities.global_definitions import (
    number_of_rows, number_of_columns,
    bit_cell_height, bit_cell_width,
    cell_brightness_threshold
)

# --- Main function ---

def decode_bitgrid(frame):

    """
    Decodes a bitgrid from the given frame by analyzing the brightness of each cell.

    Arguments:
        "frame" (np.ndarray): A NumPy array representing the frame pixels.

    Returns:
        str: A string representing the decoded bitgrid.

    """

    bits = []

    for row in range(number_of_rows): # For each row in the bitgrid:

        for column in range(number_of_columns): # For each column in the bitgrid:

            y_start = row * bit_cell_height
            y_end = y_start + bit_cell_height
            x_start = column * bit_cell_width
            x_end = x_start + bit_cell_width

            cell = frame[y_start:y_end, x_start:x_end] # Extract the cell from the frame

            average_cell_brightness = np.mean(cell) # Calculate the average brightness of the cell using NumPy mean function

            if average_cell_brightness > cell_brightness_threshold: # If the average cell brightness is above the threshold:
                bit = 1

            else: # Else (if the average cell brightness is below or equal to the threshold):
                bit = 0

            bits.append(str(bit)) # Append the bit to the list of bits

    return "".join(bits) # Return the decoded bitgrid as a string by joining the list of bits