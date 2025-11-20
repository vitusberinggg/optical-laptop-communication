# --- virtual_camera_ultra.py ---
import cv2
import threading
import time

class VideoThreadedCapture:
    """
    Ultra-optimized threaded video capture using a double-buffer model.
    - Writer thread updates a frame buffer
    - Reader retrieves instantly with NO locking delays
    """

    def __init__(self, video_path, loop=False, real_time=True):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.real_time = real_time   # If True: drop old frames
        self.loop = loop

        # Double buffer
        self.buffer_a = None
        self.buffer_b = None
        self.read_buffer = 0         # Which buffer the reader reads
        self.write_buffer = 1        # Which buffer writer writes
        self.ret = False

        # Tiny lock used only for swapping buffers
        self.swap_lock = threading.Lock()

        self.stopped = False

        # Timing
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1:
            fps = 30
        self.frame_delay = 1.0 / fps

        # Start thread
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        next_frame_time = time.time()

        while not self.stopped:
            now = time.time()
            delay = next_frame_time - now
            if delay > 0:
                time.sleep(delay)

            ret, frame = self.cap.read()
            if not ret:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Write into the inactive buffer
            if self.write_buffer == 0:
                self.buffer_a = frame
            else:
                self.buffer_b = frame

            # Swap the buffers
            with self.swap_lock:
                self.read_buffer, self.write_buffer = self.write_buffer, self.read_buffer

            self.ret = True
            next_frame_time += self.frame_delay

        self.stopped = True

    def read(self):
        """Instant frame access, zero-wait."""
        if not self.ret:
            return False, None

        # No lock needed to READ; only swap writes lock
        if self.read_buffer == 0:
            return True, self.buffer_a
        else:
            return True, self.buffer_b

    def isOpened(self):
        return not self.stopped

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()
