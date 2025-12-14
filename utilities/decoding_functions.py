
# --- Imports ---

# Library imports

import numpy as np
import time

# Non-library imports

from utilities.color_functions_bgr import tracker as tracker_bgr
from utilities.color_functions_hsv import tracker as tracker_hsv
from utilities.color_functions_hcv import tracker as tracker_hcv
from utilities.global_definitions import(
    number_of_sync_frames, number_of_rows, number_of_columns,
    bits_per_cell, bits_per_frequency, bits_per_amplitude_level,
    number_of_frequencies,
)

# --- Functions BGR ---

bits_bgr = [[[]]]

def decode_bitgrid_bgr(frame, frame_bit = 0, add_frame = False, recall = False, end_frame = False):

    """
    
    """
    
    global bits_bgr

    h, w = frame.shape[:2]
    bit_cell_height = h / number_of_rows
    bit_cell_width  = w / number_of_columns

    if add_frame:

        while len(bits_bgr) <= frame_bit:
            bits_bgr.append([])

        for row in range(number_of_rows):

            while len(bits_bgr[frame_bit]) <= row:
                bits_bgr[frame_bit].append([])

            for column in range(number_of_columns):

                while len(bits_bgr[frame_bit][row]) <= column:
                    bits_bgr[frame_bit][row].append(None)

                y0 = int(row * bit_cell_height)
                y1 = int(y0 + bit_cell_height)
                x0 = int(column * bit_cell_width)
                x1 = int(x0 + bit_cell_width)
                cell = frame[y0:y1, x0:x1]

                if end_frame:
                    bit = tracker_bgr.end_bit(row, column)

                    if bit not in ["0", "1"]:
                        bit = "0"

                    bits_bgr[frame_bit][row][column] = bit

                else:
                    tracker_bgr.add_frame(cell, row, column)

        return None

    if recall:

        collected_bytes = []
        current_byte = []

        for f in range(frame_bit):
            for row in range(number_of_rows):
                for column in range(number_of_columns):

                    value = bits_bgr[f][row][column]

                    if value is None:
                        value = "0"

                    current_byte.append(value)

                    if len(current_byte) == 8:
                        collected_bytes.append(current_byte)
                        current_byte = []

        print(f"Decoded {len(collected_bytes)} bytes from {frame_bit} frames.")

        for i, byte_bits_bgr in enumerate(collected_bytes):
            byte_str = "".join(str(b) for b in byte_bits_bgr)
            print(f"Byte {i}: {byte_str} (char: '{chr(int(byte_str,2))}')")
            
        return bits_to_message(collected_bytes)

    return None




# --- Functions HSV ---


bitgrids_hsv = []

def decode_bitgrid_hsv(hsv_frame, add_frame = False, recall = False, end_frame = False, debug_bytes = False):

    """
    Handles bitgrid collection and decoding.

    Arguments:
        hsv_frame: HSV frame for processing (only used when add_frame=True)
        add_frame: Add this frame to the tracker
        recall: Decode collected bitgrids into bytes and characters
        end_frame: Marks the end of the bit period (pushes 1 full bitgrid)

    Returns:
        str | None: Decoded message (if recall=True)

    """

    global bitgrids_hsv

    if add_frame:

        if end_frame:

            bitgrid = tracker_hsv.end_bit()

            if bitgrid is not None:
                bitgrids_hsv.append(bitgrid)

            tracker_hsv.reset()

        else:
            tracker_hsv.add_frame(hsv_frame)

        return None

    if recall:

        if len(bitgrids_hsv) == 0:
            print("No bitgrids collected yet.")
            return None

        # Combine all bitgrids horizontally
        combined = np.vstack(bitgrids_hsv)

        flat = combined.ravel()
        num_bytes = len(flat) // 8

        # Split into 8-bit chunks
        byte_matrix = flat[:num_bytes * 8].reshape(-1, 8)

        print(f"Decoded {len(byte_matrix)} bytes:")

        for i, byte_bits in enumerate(byte_matrix):

            s = "".join([ b for b in byte_bits])

            try:
                char = chr(int(s, 2))
                
            except ValueError:
                char = '?'

            if debug_bytes:
                print(f"Byte {i}: {s} (char: '{char}')")

        return bits_to_message(byte_matrix)

    return None




# --- Functions HCV ---


bitgrids_hcv = []

def decode_bitgrid_hcv(hcv_frame, add_frame = False, recall = False, end_frame = False, debug_bytes = False):

    """
    Handles bitgrid collection and decoding.

    Arguments:
        hcv_frame: HCV frame for processing (only used when add_frame=True)
        add_frame: Add this frame to the tracker
        recall: Decode collected bitgrids into bytes and characters
        end_frame: Marks the end of the bit period (pushes 1 full bitgrid)

    Returns:
        str | None: Decoded message (if recall=True)

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
            print("No bitgrids collected yet.")
            return None

        # Combine all bitgrids horizontally
        combined = np.vstack(bitgrids_hcv)

        flat = combined.ravel()
        bitstream = "".join([format(val, f"0{bits_per_cell}b") for val in flat])
        num_bytes = len(bitstream) // 8

        byte_matrix = [bitstream[i*8:(i+1)*8] for i in range(num_bytes)]

        print(f"Decoded {len(byte_matrix)} bytes:")

        for i, byte_bits in enumerate(byte_matrix):

            s = byte_bits

            try:
                char = chr(int(s, 2))
                
            except ValueError:
                char = '?'

            if debug_bytes:
                print(f"Byte {i}: {s} (char: '{char}')")

        return bits_to_message(byte_matrix)

    return None

audio_bitgrids_hcv = []

def decode_bitgrid_hcv_audio(hcv_frame, add_frame = False, recall = False, end_frame = False, debug_bytes = False):

    """
    Handles bitgrid collection and decoding for audio data.
    
    Arguments:
        "hcv_frame" (np.array): HCV frame to be processed.
        "add_frame" (bool): Boolean indicating if the frame should be added to the tracker or not.
        "recall" (bool): Boolean indicating whether it's time for the collected bitgrids to get decoded into audio data or not.
        "end_frame" (bool): Boolean that marks the end of the bit period.
        
    Returns:
        tuple | "frequency_indices", "amplitude_levels" if recall = True, else None

    """
    
    global audio_bitgrids_hcv

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
        
        bitgrids_combined = np.vstack(bitgrids_hcv) # Combines all bitgrids
        flat = bitgrids_combined.ravel()
        
        # Bitstream conversion

        values = []

        for value in flat:
            formatted_value = format(value, f"0{bits_per_cell}b")
            values.append(formatted_value)
        
        bitstream = "".join(values)
        
        if debug_bytes:
            print(f"[DEBUG] Total bits received: {len(bitstream)}")
        
        bits_per_time_frame = number_of_frequencies * (bits_per_frequency + bits_per_amplitude_level)
        number_of_complete_time_frames = len(bitstream) // bits_per_time_frame # Calculates how many complete time frames it's possible to decode
        
        print(f"\n[INFO] Decoding {number_of_complete_time_frames} time frames of audio...")
        
        frequency_indices_per_time_frame = []
        quantized_amplitude_levels_per_time_frame = []
        
        bit_position = 0
        
        for _ in range(number_of_complete_time_frames):
            
            frequency_indices = []
            amplitude_levels = []
            
            for _ in range(number_of_frequencies): # For each frequency:
                
                # Frequency bit extraction

                frequency_bits = bitstream[bit_position:bit_position + bits_per_frequency]
                bit_position += bits_per_frequency

                frequency_value = int(frequency_bits, 2)
                frequency_indices.append(frequency_value)
                
                # Amplitude bit extraction

                amp_bits = bitstream[bit_position:bit_position + bits_per_amplitude_level]
                bit_position += bits_per_amplitude_level
                
                amplitude_value = int(amp_bits, 2)
                amplitude_levels.append(amplitude_value)
            
            frequency_indices_per_time_frame.append(np.array(frequency_indices, dtype = np.int32))
            quantized_amplitude_levels_per_time_frame.append(np.array(amplitude_levels, dtype = np.int32))
        
        print(f"\n[INFO] Successfully decoded audio data for {len(frequency_indices_per_time_frame)} time frames")
        
        return frequency_indices_per_time_frame, quantized_amplitude_levels_per_time_frame
    
    return None

# --- Core worker functions ---

# Decode bitgrid
def core_decode_bitgrid_hcv(hcv_frame, end_frame = False, debug_bytes = False):

    """
    Handles bitgrid collection and decoding.

    Arguments:
        hcv_frame: HCV frame for processing (only used when add_frame=True)
        end_frame: Marks the end of the bit period (pushes 1 full bitgrid)

    Returns:
        bitgrid (tuple) | None: Decoded bitgrid (if end_frame=True)
    """

    if end_frame:

        bitgrid = tracker_hcv.end_bit()
        return bitgrid

    else:
        tracker_hcv.add_frame(hcv_frame)
        return None

# Decode message
def core_decode_message(core_bitgrid, debug_bytes=False):

    """
    Decode the message through decoded bitgrid
    
    Arguments:
        core_bitgrids_hcv (tuple)

    Returns:
        str | None: Decoded message (if len(core_bitgrid_hcv) > 0)
    """

    if len(core_bitgrid) == 0:
        print("No bitgrids collected yet.")
        return None

    # Combine all bitgrids horizontally
    combined = np.vstack(core_bitgrid)

    flat = combined.ravel()
    bitstream = "".join([format(val, f"0{bits_per_cell}b") for val in flat])
    num_bytes = len(bitstream) // 8

    byte_matrix = [bitstream[i*8:(i+1)*8] for i in range(num_bytes)]

    print(f"Decoded {len(byte_matrix)} bytes:")

    for i, byte_bits in enumerate(byte_matrix):

        s = byte_bits

        try:
            char = chr(int(s, 2))
            
        except ValueError:
            char = '?'

        if debug_bytes:
            print(f"Byte {i}: {s} (char: '{char}')")

    return bits_to_message(byte_matrix)


# --- Bit decoding functions ---

def bits_to_message(byte_matrix):

    """
    Converts a 2D list of bits (each inner list is a byte) into a readable message.

    Arguments:
        bit_matrix (list of list of int): Each inner list should contain 8 bits (0s or 1s).

    Returns:
        str: The decoded message as a string.

    """

    characters = []

    for byte_bits in byte_matrix:

        s = "".join([ b for b in byte_bits])

        try:
            characters.append(chr(int(s, 2)))

        except ValueError:
            characters.append('?')  # placeholder for invalid or partial bytes
            
    return "".join(characters)

def sync_interval_detector(color, printing = True, sync_state_dictionary = {}):

    """
    Syncs timing by detecting black/white transitions.

    Arguments:
        "color" (str):
        "detect_color_fn": Function that returns "black" or "white".
        "transitions_needed" (int): How many transitions we need to detect.
        "printing" (bool): Print debug info.
        "sync_state_dictionary" (dict): Internal persistent state across calls.

    Returns:
        "frame_interval" (float):

    """

    # Sync state dictionary initialization

    if "last_color" not in sync_state_dictionary:

        sync_state_dictionary["last_color"] = None
        sync_state_dictionary["transition_timestamps"] = []

        if printing:
            print("[SYNC] Initialized sync state dictionary, waiting for first stable color...")

    # First function call

    if sync_state_dictionary["last_color"] is None: # If this is the first function call:
        sync_state_dictionary["last_color"] = color # Store the current color in "last_color"

        if printing:
            print(f"[SYNC] Initial color = {color}")

        return 0, True # Quit the function early (no transition has occured yet)

    # Transition detection

    if color != sync_state_dictionary["last_color"]: # If the current color isn't the same as the last color:

        timestamp = time.time() # Capture the time
        sync_state_dictionary["transition_timestamps"].append(timestamp) # Save the timestamp in the sync state dictionary
        amount_of_timestamps = len(sync_state_dictionary["transition_timestamps"])

        if printing:
            print(f"[SYNC] Transition {amount_of_timestamps}: {sync_state_dictionary['last_color']} → {color}")

        sync_state_dictionary["last_color"] = color # Update "last_color"

        # Interval calculation

        if amount_of_timestamps >= (number_of_sync_frames - 1): # If the amount of timestamps is equal to or more than the amount of sync frames - 1 (the amount of transitions):

            timestamps = sync_state_dictionary["transition_timestamps"] # Get the list of timestamps

            frame_intervals = [] # Create an empty list for timestamp differences

            for timestamp_index in range (len(timestamps) - 1): # For each timestamp index in the list of timestamps:
                frame_intervals.append(timestamps[timestamp_index + 1] - timestamps[timestamp_index]) # Add the difference between that timestamp and the next one to the timestamp differences list

            average_frame_interval = sum(frame_intervals) / len(frame_intervals) # Calculate the average frame interval

            if printing:
                print(f"[SYNC] Estimated frame interval: {average_frame_interval:.4f} seconds")
                print(f"[SYNC] Timestamps in s: {frame_intervals}")

            return average_frame_interval, False

    return 0, True # If "color" = "last_color", quit (no transition detected yet)