# decoding_pipeline\audio_worker.py

import time
import numpy as np

from decoding_pipeline.shared_functions import shared as shared_class
from utilities.decoding_functions import core_decode_audio

def audio_worker(
        bitgrid_queue, 
        audio_queue, 
        stop_event, 
        last_message_timestamp, 
        debug_worker=True
    ):

    """
    Decoding worker process.

    Arguments:
        bitgrid_queue (multiprocessing.Queue): Queue of bitgrids to decode.
        audio_queue (multiprocessing.Queue): Queue of frequency indices and quantized amplitude levels.
        stop_event (multiprocessing.Event): Event to signal stop.
        last_message_timestamp (multiprocessing.Value): Timestamp of last completed message bit.
        debug_worker (bool): Enable debug prints.
    """

    frequency_indices_per_time_frame = []
    quantized_amplitude_levels_per_time_frame = []
    last_queue_debug_print = 0
    last_debug_print = 0

    assert stop_event is not None, \
        "Checks if stop event is None"

    assert bitgrid_queue is not None, \
        "Checks if bitgrid queue is None"


    while not stop_event.is_set() or not bitgrid_queue.empty():

        try:
            # Frame format: (hcv_roi, recall, add_frame, end_frame)
            msg_bitgrid, bitgrid = bitgrid_queue.get(timeout=0.1)
        except Exception:
            continue

        # --- Debug: Queue size ---
        if debug_worker:
            current_time = time.time()
            if current_time - last_queue_debug_print > 0.5:
                shared_class.log_queue("bitgrid_queue", bitgrid_queue)
                last_queue_debug_print = current_time
            decode_start = time.time()


        # --- The audio decoding ---

        if msg_bitgrid == "DATA":
            if bitgrid is None or len(bitgrid) == 0:
                continue
            decoded_data = core_decode_audio(bitgrid)
            if decoded_data is None:
                continue
            frequency_indices, amplitude_levels = decoded_data
            frequency_indices_per_time_frame.append(np.array(frequency_indices, dtype = np.int32))
            quantized_amplitude_levels_per_time_frame.append(np.array(amplitude_levels, dtype = np.int32))
        
            print(f"\n[INFO] Successfully decoded audio data for {len(frequency_indices_per_time_frame)} time frames")
            
            if debug_worker:
                print(f"\n[AUDIO] Decoded frequency indices: \n{frequency_indices}\n")
                print(f"[AUDIO] Decoded amplitude levels: \n{amplitude_levels}\n")
                
                print(f"\n[AUDIO] Frequency indices so far: \n{frequency_indices_per_time_frame}\n")
                print(f"[AUDIO] Amplitude levels so far: \n{quantized_amplitude_levels_per_time_frame}\n")

        elif msg_bitgrid == "<FLUSH>":
            audio = (frequency_indices_per_time_frame, quantized_amplitude_levels_per_time_frame)
            audio_queue.put(audio)
            print(f"[AUDIO] audio complete, queue size: {audio_queue.qsize()}")
            frequency_indices_per_time_frame = []
            quantized_amplitude_levels_per_time_frame = []   # reset ONLY after full flush

        elif msg_bitgrid == "<COMPLETE>":
            audio = (frequency_indices_per_time_frame, quantized_amplitude_levels_per_time_frame)
            audio_queue.put(audio)
            print(f"[AUDIO] audio flushed, queue size: {audio_queue.qsize()}")
            frequency_indices_per_time_frame = []
            quantized_amplitude_levels_per_time_frame = []   # reset ONLY after full message
        

        # Update timestamp for watchdog
        last_message_timestamp.value = time.time()

        # --- Debug timing ---
        if debug_worker:
            decode_end = time.time()
            if decode_end - last_debug_print > 0.5:
                print(f"[MESSAGE] Decode time: {(decode_end - decode_start) * 1000:.2f} ms")
                last_debug_print = decode_end