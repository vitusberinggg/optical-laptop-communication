
# --- Imports ---

# Library imports

import numpy as np

# Non-library imports

from utilities.global_definitions import (
    bits_per_cell, bits_per_frequency, bits_per_amplitude_level,
    number_of_frequencies
)

bitgrids_hcv = []

def decode_bitgrid_hcv_audio(hcv_frame, add_frame = False, recall = False, end_frame = False, debug_bytes = False):

    """
    Handles bitgrid collection and decoding for audio data.
    
    Arguments:
        "hcv_frame": HCV frame to be processed.
        "add_frame" (bool): Boolean indicating if the frame should be added to the tracker or not.
        "recall" (bool): Boolean indicating whether it's time for the collected bitgrids to get decoded into audio data or not.
        "end_frame" (bool): Boolean that marks the end of the bit period.
        
    Returns:
        tuple | "frequency_indices", "amplitude_levels" if recall = True, else None

    """
    
    global bitgrids_hcv

    if add_frame:

        if end_frame:

            bitgrid = tracker_hcv.end_bit()

            if bitgrid is not None:
                bitgrids_hcv.append(bitgrid)

            tracker_hcv.reset()

        else:
            tracker_hcv.add_frame(hcv_frame)

        return None
    
    if recall:
        if len(bitgrids_hcv) == 0:
            print("\n[WARNING] No bitgrids collected yet.")
            return None
        
        print(f"\n[INFO] Decoding {len(bitgrids_hcv)} bitgrids into audio data...")
        
        # Combine all bitgrids
        combined = np.vstack(bitgrids_hcv)
        flat = combined.ravel()
        
        # Convert to bitstream
        bitstream = "".join([format(val, f"0{bits_per_cell}b") for val in flat])
        
        if debug_bytes:
            print(f"[DEBUG] Total bits received: {len(bitstream)}")
        
        # Calculate how many complete time frames we can decode
        bits_per_time_frame = number_of_frequencies * (bits_per_frequency + bits_per_amplitude_level)
        number_of_complete_time_frames = len(bitstream) // bits_per_time_frame
        
        print(f"\n[INFO] Decoding {number_of_complete_time_frames} time frames of audio...")
        
        frequency_indices_per_time_frame = []
        quantized_amplitude_levels_per_time_frame = []
        
        bit_position = 0
        
        for time_frame_idx in range(number_of_complete_time_frames):
            
            frequency_indices = []
            amplitude_levels = []
            
            for freq_idx in range(number_of_frequencies):
                
                # Extract frequency bits
                freq_bits = bitstream[bit_position:bit_position + bits_per_frequency]
                bit_position += bits_per_frequency
                frequency_value = int(freq_bits, 2)
                frequency_indices.append(frequency_value)
                
                # Extract amplitude bits
                amp_bits = bitstream[bit_position:bit_position + bits_per_amplitude_level]
                bit_position += bits_per_amplitude_level
                amplitude_value = int(amp_bits, 2)
                amplitude_levels.append(amplitude_value)
            
            frequency_indices_per_time_frame.append(np.array(frequency_indices, dtype=np.int32))
            quantized_amplitude_levels_per_time_frame.append(np.array(amplitude_levels, dtype=np.int32))
        
        print(f"\n[INFO] Successfully decoded audio data for {len(frequency_indices_per_time_frame)} time frames")
        
        return frequency_indices_per_time_frame, quantized_amplitude_levels_per_time_frame
    
    return None