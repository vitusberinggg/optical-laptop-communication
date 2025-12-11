# decoding_pipeline\message_worker.py

import time

from utilities.decoding_functions import core_decode_message

def message_worker(bitgrid_queue, message_out_queue, recall, stop_flag, last_message_timestamp, debug_worker=False):
    
    """
    Decoding worker process.

    Arguments:
        bitgrid_queue (multiprocessing.Queue): Queue of bitgrids to decode.
        stop_flag (multiprocessing.Value): Boolean flag to signal stop.
        last_message_timestamp (multiprocessing.Value): Timestamp of last completed message bit.
        debug_worker (bool): Enable debug prints.
    """

    decoded_message = None
    last_queue_debug_print = 0

    while not stop_flag.value or not bitgrid_queue.empty():

        try:
            # Frame format: (hcv_roi, recall, add_frame, end_frame)
            bitgrid = bitgrid_queue.get(timeout=0.1)
        except Exception:
            continue

        # --- Debugging ---

        if debug_worker:
            current_time = time.time()
            if current_time - last_queue_debug_print > 0.5:
                print(f"[DEBUG] Decode worker queue size = {bitgrid_queue.qsize()}")
                last_queue_debug_print = current_time
            decode_start_time = time.time()

        # --- Decode message ---
        
        if len(bitgrid) > 0:
            decoded_message.join(core_decode_message(bitgrid))

        if recall:
            final_message = "".join(decoded_message)
            message_out_queue.put(final_message)
            decoded_message = []   # reset for next message

        # Update timestamp for watchdog
        last_message_timestamp.value = time.time()

        # --- Debugging timing ---

        if debug_worker:
            decode_end_time = time.time()
            if decode_end_time - last_queue_debug_print > 0.5:
                print(f"[DEBUG] Decode time: {(decode_end_time - decode_start_time) * 1000:.2f} ms")
        

