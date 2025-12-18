# decoding_pipeline\decoder_worker.py

import time
import queue
import numpy as np
import multiprocessing
from multiprocessing import queues
from utilities.decoding_functions import core_decode_bitgrid_hcv
from utilities.color_functions_hcv import tracker, bitgrid_majority_calculator as numba, dominant_color_hcv
from decoding_pipeline.shared_functions import shared_class

def decoding_worker(
        frame_queue, 
        command_queue, 
        bitgrid_queue, 
        stop_event, 
        last_decode_timestamp, 
        debug_worker=False
    ):
    
    """
    Decoding worker process.

    Arguments:
        frame_queue (multiprocessing.Queue): Queue of frames to decode.
        command_queue (multiprocessing.Queue): Queue of LUT and color names
        bitgrid_queue (multiprocessing.Queue): Queue of bitgrids that is decoded
        stop_event (multiprocessing.Event): Event to signal stop.
        last_decode_timestamp (multiprocessing.Value): Timestamp of last completed decode.
        debug_worker (bool): Enable debug prints.
    """

    last_queue_debug_print = 0
    last_timing_debug_print = 0
    LUT_ready = False
    bitgrid = [[]]  # Initialize bitgrid as empty list

    assert stop_event is not None, "stop_event must be provided"
    assert frame_queue is not None, "frame_queue must be provided"
    assert command_queue is not None, "command_queue must be provided"
    assert bitgrid_queue is not None, "bitgrid_queue must be provided"

    while not stop_event.is_set() or not frame_queue.empty():

        
        # Check for commands
        try:
            cmd, payload = command_queue.get_nowait()

            if cmd == "set_lut":
                tracker.LUT, tracker.color_names = payload
                LUT_ready = True
                print("[WORKER] LUT received and initialized.")
                warmup_all()
                print("[WORKER] numba pre-compiled.")

            elif cmd == "shutdown":
                print("[Worker] Shutdown received.")
                break

        except (queue.Empty, queues.Empty):
            pass

        # Don't decode until LUT exists
        if not LUT_ready:
            time.sleep(0.01)
            continue

        try:
            # Frame format: (hcv_roi, recall, add_frame, end_frame)
            hcv_roi, add_frame, end_frame = frame_queue.get(timeout=0.1)
        except Exception:
            continue

        # --- Debugging ---
        if debug_worker:
            current_time = time.time()
            if current_time - last_queue_debug_print > 0.5:
                print(f"[WORKER] Decode worker queue size = {frame_queue.qsize()}")
                last_queue_debug_print = current_time
            decode_start_time = time.time()

        # --- Decode frame ---

        color = dominant_color_hcv(hcv_roi)
        
        if color != "orange":
            bitgrid = core_decode_bitgrid_hcv(hcv_roi, end_frame, debug_bytes=False)
        else:
            bitgrid_queue.put(("<COMPLETE>", None))
            continue

        # --- Skip invalid results ---
        if bitgrid is None or (isinstance(bitgrid, np.ndarray) and bitgrid.size == 0):
            continue

        # --- Push into queue ---
        try:
            bitgrid_queue.put(("DATA", bitgrid), timeout=0.1)
        except queue.Full:
            print("[WARNING] Bitgrid queue is full.")

        # Update timestamp for watchdog
        last_decode_timestamp.value = time.time()

        # --- Debugging timing ---
        if debug_worker:
            decode_end_time = time.time()
            decode_time_ms = (decode_end_time - decode_start_time) * 1000.0

            # print only every 0.5 sec
            if decode_end_time - last_timing_debug_print > 0.5:
                print(f"[WORKER] Decode time: {decode_time_ms:.2f} ms")
                last_timing_debug_print = decode_end_time

def warmup_all():

    """
    Performs a one-time warm-up for "bitgrid_majority_calculator" by calling it with a dummy input.
    """

    dummy_array = np.zeros((2, 2, 8, 16, 10), dtype = np.uint8)
    numba(dummy_array, 5)