# decoding_pipeline/watchdog.py

import time

def watchdog(
        last_decode_timestamp, 
        stop_flag, 
        watchdog_on=False, 
        stall_threshold=1.0
    ):
    
    """
    Watchdog process to detect decoding pipeline stalls.

    Arguments:
        last_decode_timestamp (multiprocessing.Value): Timestamp of last completed decode.
        stop_flag (multiprocessing.Value): Boolean flag to signal stop.
        watchdog_on (bool): Enable or disable watchdog monitoring.
        stall_threshold (float): Time in seconds to consider the pipeline stalled.
    """

    while watchdog_on and not stop_flag.value:
        
        try:
            time_since_last_decode = time.time() - last_decode_timestamp.value
            if time_since_last_decode > stall_threshold:
                print(f"[WARNING] Decode thread stalled! No frames processed for {time_since_last_decode:.2f} seconds.")
        except Exception as e:
            print(f"[ERROR] Watchdog exception: {e}")
        
        time.sleep(0.2)  # Check 5 times per second
