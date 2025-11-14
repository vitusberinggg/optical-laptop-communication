
# --- Imports ---

import numpy as np
import cv2

from utilities.global_definitions import number_of_rows, number_of_columns, bit_cell_height, bit_cell_width, cell_brightness_threshold

# --- Functions ---

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

def bits_to_message(bits):

    """
    Converts a string of bits into a readable message.
    
    Arguments:
        "bits" (str): A string representing the bits to be converted.
        
    Returns:
        str: A string representing the decoded message.
        
    """

    characters = []

    for bit_index in range(0, len(bits), 8): # For each byte (8 bits) in the bit string:

        byte = bits[bit_index:bit_index + 8] # Extract the byte using slicing

        if len(byte) < 8: # If the length of the byte is less than 8 bits:
            continue # Skip it

        characters.append(chr(int(byte, 2))) # Convert the byte to a character and append it to the list

    return "".join(characters) # Return the decoded message by joining the list of characters into a string

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
        counts = {
            "white": int(cv2.countNonZero(white_mask)),
            "black": int(cv2.countNonZero(black_mask)),
            "blue": int(cv2.countNonZero(blue_mask)),
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

        if color == "blue":
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