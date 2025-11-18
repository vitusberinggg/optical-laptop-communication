
# --- Imports ---
from collections import Counter
import color_functions
import numpy as np
import cv2

from utilities.global_definitions import number_of_rows, number_of_columns, bit_cell_height, bit_cell_width, cell_brightness_threshold

# --- Functions ---

def decode_bitgrid(frame, frame_bit=1, recall=False, end_frame=False):

    """
    Decodes a bitgrid from the given frame by analyzing the brightness of each cell.

    Arguments:
        "frame" (np.ndarray): A NumPy array representing the frame pixels.

    Returns:
        str: A string representing the decoded bitgrid.

    """

    h, w = frame.shape[:2]

    bit_cell_height = h/number_of_rows
    bit_cell_width = w/number_of_columns

    bits = [[[]]]
    bytes = [[]]
    byte = []

    for row in range(number_of_rows): # For each row in the bitgrid:

        for column in range(number_of_columns): # For each column in the bitgrid:
            
                # Expand rows if necessary
            while len(bits) < frame_bit:
                bits.append([])
                # Expand rows if necessary
            while len(bits) < row:
                bits.append([])

            y_start = row * bit_cell_height
            y_end = y_start + bit_cell_height
            x_start = column * bit_cell_width
            x_end = x_start + bit_cell_width

            cell = frame[y_start:y_end, x_start:x_end] # Extract the cell from the frame

            if end_frame: # If it's the end of the frame:
                value = color_functions.tracker.end_bit(row, column)  # Finalize the previous bit in the color tracker

                bits[frame_bit][row].append(value)  # Append the finalized bit value to the bits list
            else:
                color_functions.tracker.add_frame(cell, row, column)  # Add the cell to the color tracker

    if recall:
        

        for f in range (frame_bit):
            for row in range (number_of_rows):
                for column in range (number_of_columns):

                    byte.append(bits[f][number_of_rows][number_of_columns])

                    if byte[:8]:
                        bytes.append(byte)
                        byte = []
                    elif len(byte) >= 8:
                        byte = []

        message = bits_to_message(bytes)
    
    return message

def bits_to_message(bit_matrix):
    """
    Converts a 2D list of bits (each inner list is a byte) into a readable message.

    Arguments:
        bit_matrix (list of list of int): Each inner list should contain 8 bits (0s or 1s).

    Returns:
        str: The decoded message as a string.
    """
    characters = []

    for byte_bits in bit_matrix:
        if len(byte_bits) != 8:
            continue  # skip invalid bytes
        byte_str = "".join(str(b) for b in byte_bits)
        characters.append(chr(int(byte_str, 2)))

    return "".join(characters)

def decode_bits_with_blue(frames, roi_size=100, verbose=False):
    """
    Decodes bits from a sequence of frames where each bit is a colored frame:
    - white: bit 1
    - black: bit 0
    - blue: separator (next bit ready)
    
    Arguments:
        frames (list of np.ndarray): List of frames (BGR) from the sender.
        roi_size (int): Size of the square region to read the color from.
        verbose (bool): Print debug info.
    
    Returns:
        str: Binary string representing the decoded bits.
    """
    def read_color(frame):
        """Detects dominant color in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0,0,200), (180,30,255))
        black_mask = cv2.inRange(hsv, (0,0,0), (180,255,50))
        blue_mask  = cv2.inRange(hsv, (100,150,0), (140,255,255))
        green_mask = cv2.inRange(hsv, (45,80,80), (75,255,255))
        counts = {
            "white": int(cv2.countNonZero(white_mask)),
            "black": int(cv2.countNonZero(black_mask)),
            "blue": int(cv2.countNonZero(blue_mask)),
            "green": int(cv2.countNonZero(green_mask)),
        }
        return max(counts, key=counts.get)

    bits = ""
    last_color = None
    bit_ready = True  # ready for first bit

    for frame in frames:
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        roi = frame[max(0,cy-roi_size):min(h,cy+roi_size), max(0,cx-roi_size):min(w,cx+roi_size)]
        color = read_color(roi)

        if color in ["blue", "green"]:
            bit_ready = True
            if verbose:
                print("Blue separator detected — next bit ready.")
        elif color in ["white","black"] and bit_ready:
            bits += "1" if color == "white" else "0"
            if verbose:
                print(f"Decoded bit: {bits[-1]}")
            bit_ready = False  # wait for next blue separator

        last_color = color

    return bits
