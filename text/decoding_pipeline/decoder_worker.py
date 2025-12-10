# decoding_pipeline/decoder_worker.py

import time
from utilities.decoding_functions import decode_bitgrid_hcv

def decoding_worker(frame_queue, stop_flag, last_decode_timestamp, debug_worker=False):
    """
    Decoding worker process.

    Arguments:
        frame_queue (multiprocessing.Queue): Queue of frames to decode.
        stop_flag (multiprocessing.Value): Boolean flag to signal stop.
        last_decode_timestamp (multiprocessing.Value): Timestamp of last completed decode.
        debug_worker (bool): Enable debug prints.
    """

    decoded_message = None
    last_queue_debug_print = 0

    while not stop_flag.value or not frame_queue.empty():
        try:
            # Frame format: (hcv_roi, recall, add_frame, end_frame)
            hcv_roi, recall, add_frame, end_frame = frame_queue.get(timeout=0.1)
        except Exception:
            continue

        # --- Debugging ---
        if debug_worker:
            current_time = time.time()
            if current_time - last_queue_debug_print > 0.5:
                print(f"[DEBUG] Decode worker queue size = {frame_queue.qsize()}")
                last_queue_debug_print = current_time
            decode_start_time = time.time()

        # --- Decode frame ---
        if recall:            
            result = decode_bitgrid_hcv(hcv_roi, add_frame, recall, end_frame, debug_bytes=False)
            if isinstance(result, str) and result.strip():
                decoded_message = result
            
        else:
            decode_bitgrid_hcv(hcv_roi, add_frame, recall, end_frame, debug_bytes=False)

        # Update timestamp for watchdog
        last_decode_timestamp.value = time.time()

        # --- Debugging timing ---
        if debug_worker:
            decode_end_time = time.time()
            if decode_end_time - last_queue_debug_print > 0.5:
                print(f"[DEBUG] Decode time: {(decode_end_time - decode_start_time) * 1000:.2f} ms")
