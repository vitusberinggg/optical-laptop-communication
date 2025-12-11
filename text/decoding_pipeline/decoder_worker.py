# decoding_pipeline\decoder_worker.py

import time
import queue
import multiprocessing
from multiprocessing import queues
from utilities.decoding_functions import core_decode_bitgrid_hcv
from utilities.color_functions_hcv import tracker
from decoding_pipeline.shared_functions import shared_class

def decoding_worker(frame_queue, command_queue, bitgrid_queue, stop_flag, last_decode_timestamp, debug_worker=False):
    """
    Decoding worker process.

    Arguments:
        frame_queue (multiprocessing.Queue): Queue of frames to decode.
        command_queue (multiprocessing.Queue): Queue of LUT and color names
        stop_flag (multiprocessing.Value): Boolean flag to signal stop.
        last_decode_timestamp (multiprocessing.Value): Timestamp of last completed decode.
        debug_worker (bool): Enable debug prints.
    """

    last_queue_debug_print = 0
    LUT_ready = False
    bitgrid = [[]]  # Initialize bitgrid as empty list

    while not stop_flag.value or not frame_queue.empty():

        
        # Check for commands
        try:
            cmd, payload = command_queue.get_nowait()

            if cmd == "set_lut":
                tracker.LUT, tracker.color_names = payload
                LUT_ready = True
                print("[Worker] LUT received and initialized.")

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
                print(f"[DEBUG] Decode worker queue size = {frame_queue.qsize()}")
                last_queue_debug_print = current_time
            decode_start_time = time.time()

        # --- Decode frame ---
        
        if end_frame:
            bitgrid = core_decode_bitgrid_hcv(hcv_roi, end_frame, debug_bytes=False)
        elif add_frame:
            core_decode_bitgrid_hcv(hcv_roi, end_frame, debug_bytes=False)

        if len(bitgrid) == 0:
            continue  # Skip if no bitgrid was produced
        else:
            try:
                bitgrid_queue.put(bitgrid)
            except multiprocessing.queues.full():
                print("[WARNIG] It's full you jackass...")

        # Update timestamp for watchdog
        last_decode_timestamp.value = time.time()

        # --- Debugging timing ---
        if debug_worker:
            decode_end_time = time.time()
            if decode_end_time - last_queue_debug_print > 0.5:
                print(f"[DEBUG] Decode time: {(decode_end_time - decode_start_time) * 1000:.2f} ms")
