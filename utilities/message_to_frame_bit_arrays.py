
# --- Imports ---

import numpy as np

from global_definitions import number_of_columns, number_of_rows

# --- Main function ---

def message_to_frame_bit_arrays(message):

    """
    Turns a message into a list of frames represented as 2D NumPy arrays of bits.

    Arguments:
        "message" (str): The message to be converted.

    Returns:
        "frame_bit_arrays": A list of 2D NumPy arrays representing the frames.
    
    """

    binary_list = []

    for character in message: # For each character in the message:
        ascii_value = ord(character) # Convert it to ASCII
        binary_string = format(ascii_value, "08b") # Format the value as an 8-bit binary string
        binary_list.append(binary_string) # Add the string to the binary list
    
    bits = "".join(binary_list) # Merge all strings in the binary list into a single string

    frame_capacity = number_of_rows * number_of_columns
    frame_bit_arrays = []

    for start_index in range(0, len(bits), frame_capacity): # For each starting index in the range 0 - len(bits), stepping with the frame capacity:

        chunk = bits[start_index:start_index + frame_capacity] # Slice a chunk the size of the frame capacity from the string of bits

        if len(chunk) < frame_capacity: # If the length of the chunk is smaller than the frame capacity
            chunk = chunk.ljust(frame_capacity, "0") # Pad with 0's until the chunk is the same size as the frame capacity

        frame_array = np.array(list(chunk), dtype = np.uint8).reshape((number_of_rows, number_of_columns)) # Convert the chunk to a list, convert the list into an NumPy array (more efficient) and then reshape the one-dimensional array into a 2D array

        frame_bit_arrays.append(frame_array) # Add the frame array into the list of frames

    return frame_bit_arrays