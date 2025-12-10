
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
        "visual_frames" (list of np.arrays)

    """

    # Bit array list creation

    frame_bit_arrays = []

    for time_frame_index in range(len(frequency_indices_per_time_frame)): # For each time frame:
        
        frequency_indices = frequency_indices_per_time_frame[time_frame_index]

        amplitude_levels = quantized_amplitude_levels_per_time_frame[time_frame_index]

        for frequency_index in range(len(frequency_indices)): # For each frequency:

            frequency_bits = format(int(frequency_indices[frequency_index]), f'0{bits_per_frequency}b')

            for bit in frequency_bits: # For each frequency bit:
                frame_bit_arrays.extend(int(bit))
            
            amplitude_bits = format(int(amplitude_levels[frequency_index]), f'0{bits_per_amplitude_level}b')

            for bit in amplitude_bits: # For each amplitude bit:
                frame_bit_arrays.extend(int(bit))

    # Zero-padding (to make "frame_bit_arrays" divisible by "bits_per_cell")

    remainder = len(frame_bit_arrays) % bits_per_cell

    if remainder != 0:
        padding = bits_per_cell - remainder
        frame_bit_arrays.extend([0] * padding)
    
    # Frame bit arrays to cell values conversion

    cell_values = []

    for i in range(0, len(frame_bit_arrays), bits_per_cell):

        cell_bits = frame_bit_arrays[i:i + bits_per_cell]

        cell_value = int(''.join(map(str, cell_bits)), 2)

        cell_values.append(cell_value)

    # Cell values to visual frames conversion

    visual_frames = []

    for i in range(0, len(cell_values), number_of_cells):

        frame_cells = cell_values[i:i + number_of_cells]

        if len(frame_cells) < number_of_cells:
            frame_cells.extend([0] * (number_of_cells - len(frame_cells)))

        visual_frames.append(np.array(frame_cells, dtype = np.uint8))
    
    return visual_frames