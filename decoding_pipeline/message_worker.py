# decoding_pipeline\message_worker.py

import time

from utilities.decoding_functions import core_decode_message
from decoding_pipeline.shared_functions import shared_class

def message_worker(
        bitgrid_queue, 
        message_queue, 
        stop_event, 
        last_message_timestamp, 
        debug_worker=True
    ):
    
    """
    Decoding worker process.

    Arguments:
        bitgrid_queue (multiprocessing.Queue): Queue of bitgrids to decode.
        stop_event (multiprocessing.Event): Event to signal stop.
        last_message_timestamp (multiprocessing.Value): Timestamp of last completed message bit.
        debug_worker (bool): Enable debug prints.
    """

    message_buffer = ""   # accumulate full message here
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


        if msg_bitgrid == "DATA":
            if bitgrid is None or len(bitgrid) == 0:
                continue
            block = "".join(core_decode_message(bitgrid))
            message_buffer += block   # <-- append to cumulative message
            if debug_worker:
                print(f"[MESSAGE] Decoded block: {block}")
                print(f"[MESSAGE] Message buffer so far: {message_buffer}")

        elif msg_bitgrid == "<FLUSH>":
            message_queue.put(message_buffer)
            print(f"[Message] message complete, queue size: {message_queue.qsize()}")
            message_buffer = ""   # reset ONLY after full flush

        elif msg_bitgrid == "<COMPLETE>":
            message_queue.put(message_buffer)
            print(f"[Message] message flushed, queue size: {message_queue.qsize()}")
            message_buffer = ""   # reset ONLY after full message
        

        # Update timestamp for watchdog
        last_message_timestamp.value = time.time()

        # --- Debug timing ---
        if debug_worker:
            decode_end = time.time()
            if decode_end - last_debug_print > 0.5:
                print(f"[MESSAGE] Decode time: {(decode_end - decode_start) * 1000:.2f} ms")
                last_debug_print = decode_end
        

