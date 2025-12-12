
# --- Imports ---

import numpy as np

from utilities.global_definitions import(
    number_of_columns, number_of_rows,
    bits_per_frequency, bits_per_amplitude_level,
    bits_per_cell, number_of_cells
)

# --- Functions ---

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

def message_to_frame_several_bit_arrays(message, bits_per_cell):

    binary_string = "".join([format(ord(character), "08b") for character in message])

    chunks = [binary_string[i:i+bits_per_cell] for i in range(0, len(binary_string), bits_per_cell)]

    cell_values = [int(chunk, 2) for chunk in chunks]

    frame_capacity = number_of_rows * number_of_columns

    frame_bit_arrays = []

    for start_idx in range(0, len(cell_values), frame_capacity):
        frame_chunk = cell_values[start_idx:start_idx + frame_capacity]

        if len(frame_chunk) < frame_capacity:
            frame_chunk += [0] * (frame_capacity - len(frame_chunk))
            
        frame_array = np.array(frame_chunk, dtype=np.uint8).reshape((number_of_rows, number_of_columns))
        frame_bit_arrays.append(frame_array)

    return frame_bit_arrays

def audio_data_to_frame_bit_arrays(frequency_indices_per_time_frame, quantized_amplitude_levels_per_time_frame):

    """
    Converts compressed audio data into bit arrays.

    Arguments:
        "frequency_indices_per_time_frame" (list of np.arrays): List where each element contains the indices of the loudest frequencies for that time frame.
        "quantized_amplitude_levels_per_time_frame" (list of np.arrays): List where each element contains the quantized amplitude levels corresponding to the frequency indices.

    Returns:
        "visual_frames" (list of np.arrays): List where each element represents a visual frame.

    """

    # Bit array list creation

    frame_bit_arrays = []

    for time_frame_index in range(len(frequency_indices_per_time_frame)): # For each time frame:
        
        frequency_indices = frequency_indices_per_time_frame[time_frame_index]

        amplitude_levels = quantized_amplitude_levels_per_time_frame[time_frame_index]

        for frequency_index in range(len(frequency_indices)): # For each frequency:

            frequency_bits = format(int(frequency_indices[frequency_index]), f'0{bits_per_frequency}b')

            for bit in frequency_bits: # For each frequency bit:
                frame_bit_arrays.append(int(bit))
            
            amplitude_bits = format(int(amplitude_levels[frequency_index]), f'0{bits_per_amplitude_level}b')

            for bit in amplitude_bits: # For each amplitude bit:
                frame_bit_arrays.append(int(bit))

    # Zero-padding (to make "frame_bit_arrays" divisible by "bits_per_cell")

    remainder = len(frame_bit_arrays) % bits_per_cell

    if remainder != 0:
        padding = bits_per_cell - remainder
        frame_bit_arrays.extend([0] * padding)
    
    # Frame bit arrays to cell values conversion

    cell_values = []

    for bit_array_index in range(0, len(frame_bit_arrays), bits_per_cell): # Iterate over "frame_bit_array" in steps of "bits_per_cell" (each iteration processes one "cell" worth of bits)

        cell_bits_slice = frame_bit_arrays[bit_array_index:bit_array_index + bits_per_cell] # Extracts a slice of "bits_per_cell" consecutive bits from the bit array

        cell_bits_string_list = map(str, cell_bits_slice) # Converts each bit (integer) to a string

        cell_bits_string = "".join(cell_bits_string_list) # Converts the list of strings into one string

        cell_value = int(cell_bits_string, 2) # Converts the binary string into a decimal integer with the base 2 (example: "101" --> 5)

        cell_values.append(cell_value)

    # Cell values to visual frames conversion

    visual_frames = []

    for cell_index in range(0, len(cell_values), number_of_cells): # Loops through cell values in steps of "number_of_cells" (each iteration creates one visual frame)

        frame_cells = cell_values[cell_index:cell_index + number_of_cells] # Extracts a slice of "number_of_cells" consecutive values from "cell_values"

        if len(frame_cells) < number_of_cells: # If there aren't enough cells (a last, partial frame)
            frame_cells.extend([0] * (number_of_cells - len(frame_cells))) # Pad it with zeros

        frame_cell_array = np.array(frame_cells, dtype = np.uint8).reshape(number_of_rows, number_of_columns) # Convert "frame_cells" into a NumPy array

        visual_frames.append(frame_cell_array)
    
    return visual_frames