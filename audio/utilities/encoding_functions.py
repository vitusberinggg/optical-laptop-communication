
# --- Imports ---

# Library imports

import numpy as np

# Non-library imports

from global_definitions import(
    bits_per_frequency, bits_per_amplitude_level, bits_per_cell,
    number_of_cells
)

# --- Functions ---

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

        frame_cell_array = np.array(frame_cells, dtype = np.uint8) # Convert "frame_cells" into a NumPy array

        visual_frames.append(frame_cell_array)
    
    return visual_frames