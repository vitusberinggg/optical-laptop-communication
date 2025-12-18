# decoding_pipeline\shared_functions.py

import multiprocessing
from multiprocessing import shared_memory
import time
import queue
import numpy as np

class shared:

    def __init__(self):
        # Shared objects
        self._frame_queue = None
        self._command_queue = None
        self._bitgrid_queue = None
        self._message_queue = None
        self._audio_queue = None

        self._stop_event = None

        self._recall_flag = None

        self._last_decode_timestamp = None
        self._last_message_timestamp = None
        self._last_frame = None

        self.shm = None
        self.shm_array = None
        self.dtype = np.uint8


    def initialize_shared_objects(self, queue_maxsize=100):
        """
        Initializes shared objects for the decoding pipeline.
        """
        self._frame_queue = multiprocessing.Queue(maxsize=queue_maxsize)
        self._command_queue = multiprocessing.Queue(maxsize=queue_maxsize)
        self._bitgrid_queue = multiprocessing.Queue(maxsize=queue_maxsize)
        self._message_queue = multiprocessing.Queue(maxsize=queue_maxsize)
        self._audio_queue = multiprocessing.Queue(maxsize=queue_maxsize)

        self._stop_event = multiprocessing.Event()

        self._recall_flag = multiprocessing.Value('b', False)
        self._last_frame = multiprocessing.Value('b', False)

        self._last_decode_timestamp = multiprocessing.Value('d', time.time())  # double timestamp
        self._last_message_timestamp = multiprocessing.Value('d', time.time())
        print("[Shared] Shared objects initialized.")
    

    def get_shared_objects(self):
        """
        Returns the shared objects for the decoding pipeline.
        """
        return (
            self._frame_queue, self._command_queue, self._bitgrid_queue,
            self._message_queue, self._audio_queue, 
            self._stop_event, self._last_frame, self._recall_flag,
            self._last_decode_timestamp, self._last_message_timestamp
        )
    
    def log_queue(self, name, q):
        """
        Logs the size and contents of a queue WITHOUT modifying it.
        Works with queue.Queue and most queue-like objects.
        """
        try:
            # Most queue.Queue objects expose ".queue" (deque)
            contents = list(q.queue)
        except Exception:
            # Fallback: don't crash logger
            contents = "<unavailable>"

        print(f"[DEBUG] {name} | size={q.qsize()} | contents={contents}")
    

    # --- Decoding worker ---

    def preallocate_shared_memory(self, frame):
        frame_shape = frame.shape
        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(frame_shape) * np.dtype(self.dtype).itemsize)
        self.shm_array = np.ndarray(frame_shape, dtype=self.dtype, buffer=self.shm.buf)

    def push_frame(self, frame_data):
        """
        Push a frame into the shared decoding queue.

        Arguments:
            frame_data (tuple): (hcv_roi, add_frame, end_frame)
        """

        if self._frame_queue is None:
            raise RuntimeError("Pipeline not started. Call start_pipeline() first.")

        '''
        np.copyto(self.shm_array, hcv_roi)

        queue_item = (self.shm.name, self.shm_array.shape, str(self.shm_array.dtype), add_frame, end_frame)
        '''
        try:
            self._frame_queue.put(frame_data, timeout=0.1)
        except multiprocessing.queues.Full:
            # Optional: drop frame if queue is full
            pass

    def push_LUT(self, LUT, color_names):
        """Send LUT to worker at any time after startup."""
        self._command_queue.put(("set_lut", (LUT, color_names)))
        print("[Pipeline] LUT pushed to worker.")


    # --- Message worker ---

    def pull_decoded_message(self, max_wait=0.5):
        """
        Pull a decoded message from the message worker without blocking the GUI.
        
        Arguments:
            max_wait (float): Maximum time in seconds to wait for a message.
        """

        self._bitgrid_queue.put(("<FLUSH>", None))
        
        start_time = time.time()

        decoded_message = None
        while time.time() - start_time < max_wait:
            try:
                decoded_message = self._message_queue.get_nowait()
                break  # message received
            except queue.Empty:
                time.sleep(0.01)  # yield CPU / allow GUI to check for input

        
        return decoded_message
    

    # --- Audio worker ---

    def pull_decoded_audio_data(self, max_wait=0.5):
        """
        Pull a decoded audio data from the message worker without blocking the GUI.
        
        Arguments:
            max_wait (float): Maximum time in seconds to wait for audio data.
        """

        self._bitgrid_queue.put(("<FLUSH>", None))
        
        start_time = time.time()

        decoded_audio_data = None
        while time.time() - start_time < max_wait:
            try:
                decoded_audio_data = self._audio_queue.get_nowait()
                break  # message received
            except queue.Empty:
                time.sleep(0.01)  # yield CPU / allow GUI to check for input

        
        return decoded_audio_data
    

shared_class = shared()
