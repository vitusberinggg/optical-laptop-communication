# decoding_pipeline/pipeline.py

import multiprocessing
import psutil
import time
from decoding_pipeline.decoder_worker import decoding_worker
from decoding_pipeline.watchdog import watchdog

# Shared objects
_frame_queue = None
_stop_flag = None
_last_decode_timestamp = None

# Processes
_decode_process = None
_watchdog_process = None


def start_pipeline(core_worker=None, core_watchdog=None, queue_maxsize=100):
    """
    Starts the decoding worker and watchdog processes.

    Arguments:
        core_worker (list[int] | None): CPU cores for decoding worker.
        core_watchdog (list[int] | None): CPU cores for watchdog.
        queue_maxsize (int): Maximum size of the frame queue.
    """

    global _frame_queue, _stop_flag, _last_decode_timestamp
    global _decode_process, _watchdog_process

    # Shared objects
    _frame_queue = multiprocessing.Queue(maxsize=queue_maxsize)
    _stop_flag = multiprocessing.Value('b', False)  # boolean stop flag
    _last_decode_timestamp = multiprocessing.Value('d', time.time())  # double timestamp

    # Start decoding worker process
    _decode_process = multiprocessing.Process(
        target=decoding_worker,
        args=(_frame_queue, _stop_flag, _last_decode_timestamp),
        daemon=True
    )
    _decode_process.start()

    # Pin decoding worker to specific cores
    if core_worker:
        try:
            if isinstance(core_worker, int):
                core_worker = [core_worker]  # Convert single core to list
            else:
                core_worker = list(core_worker)  # Ensure it's a list
            psutil.Process(_decode_process.pid).cpu_affinity(core_worker)
        except Exception as e:
            print(f"[WARNING] Could not pin decoding worker cores: {e}")

    # Start watchdog process
    _watchdog_process = multiprocessing.Process(
        target=watchdog,
        args=(_last_decode_timestamp, _stop_flag),
        daemon=True
    )
    _watchdog_process.start()

    # Pin watchdog to specific cores
    if core_watchdog:
        try:
            if isinstance(core_watchdog, int):
                core_watchdog = [core_watchdog]  # Convert single core to list
            else:
                core_watchdog = list(core_watchdog)  # Ensure it's a list
            psutil.Process(_watchdog_process.pid).cpu_affinity(core_watchdog)
        except Exception as e:
            print(f"[WARNING] Could not pin watchdog cores: {e}")


def stop_pipeline():
    """
    Stops decoding worker and watchdog processes cleanly.
    """
    global _stop_flag, _decode_process, _watchdog_process

    if _stop_flag is None:
        return

    # Signal stop
    _stop_flag.value = True

    # Wait for processes to finish
    if _decode_process:
        _decode_process.join(timeout=5)
    if _watchdog_process:
        _watchdog_process.join(timeout=5)


def push_frame(frame_data):
    """
    Push a frame into the shared decoding queue.

    Arguments:
        frame_data (tuple): (hcv_roi, recall, add_frame, end_frame)
    """
    global _frame_queue
    if _frame_queue is None:
        raise RuntimeError("Pipeline not started. Call start_pipeline() first.")

    try:
        _frame_queue.put(frame_data, timeout=0.1)
    except multiprocessing.queues.Full:
        # Optional: drop frame if queue is full
        pass
