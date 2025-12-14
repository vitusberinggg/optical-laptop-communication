# decoding_pipeline\message_worker.py

import time

from utilities.decoding_functions import core_decode_message
from decoding_pipeline.shared_functions import shared_class

def message_worker(bitgrid_queue, message_queue, recall, last_frame, stop_flag, last_message_timestamp, debug_worker=False):
    
    """
    Decoding worker process.

    Arguments:
        bitgrid_queue (multiprocessing.Queue): Queue of bitgrids to decode.
        stop_flag (multiprocessing.Value): Boolean flag to signal stop.
        last_message_timestamp (multiprocessing.Value): Timestamp of last completed message bit.
        debug_worker (bool): Enable debug prints.
    """

    message_buffer = ""   # accumulate full message here
    last_queue_debug_print = 0
    last_debug_print = 0

    while not stop_flag.value or not bitgrid_queue.empty():

        try:
            # Frame format: (hcv_roi, recall, add_frame, end_frame)
            bitgrid = bitgrid_queue.get(timeout=0.1)
        except Exception:
            continue

        # --- Debug: Queue size ---
        if debug_worker:
            current_time = time.time()
            if current_time - last_queue_debug_print > 0.5:
                shared_class.log_queue("bitgrid_queue", bitgrid_queue)
                last_queue_debug_print = current_time
            decode_start = time.time()

        # --- Decode message block ---
        if len(bitgrid) > 0:
            block = "".join(core_decode_message(bitgrid))
            message_buffer += block   # <-- append to cumulative message
            if debug_worker:
                print(f"[MESSAGE] Decoded block: {block}")
                print(f"[MESSAGE] Message buffer so far: {message_buffer}")

        # --- Flush output only if recall=True ---
        if recall and last_frame.value:
            message_queue.put(message_buffer)
            print(f"[Message] message queue size: {message_queue.qsize()}")
            message_buffer = ""   # reset ONLY after full flush
        

        # Update timestamp for watchdog
        last_message_timestamp.value = time.time()

        # --- Debug timing ---
        if debug_worker:
            decode_end = time.time()
            if decode_end - last_debug_print > 0.5:
                print(f"[MESSAGE] Decode time: {(decode_end - decode_start) * 1000:.2f} ms")
                last_debug_print = decode_end
        

