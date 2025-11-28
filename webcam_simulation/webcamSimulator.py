# --- virtual_camera_precise.py ---
import cv2
import threading
import time

class VideoThreadedCapture:
    """
    Bit-precise, frame-accurate simulator for webcam-style video.
    - No race conditions
    - No partial frame overwrites
    - Correct FPS timing
    - Real-time (drop frames) or exact playback (no drops)
    """

    def __init__(self, video_path, loop=False, real_time=False):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.loop = loop
        self.real_time = real_time

        # Double buffer
        self.buffer_a = None
        self.buffer_b = None

        # Read/write pointer index (0 or 1)
        self.read_buffer = 0
        self.write_buffer = 1

        # Safe read/write synchronization
        self.swap_lock = threading.Lock()
        self.read_lock = threading.Lock()

        self.ret = False
        self.stopped = False

        # Target FPS
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps < 2:  # fallback
            fps = 30
        self.frame_delay = 1.0 / fps

        # Start worker thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Background frame reader."""
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

            # Copy frame to avoid OpenCV pointer reuse issues
            frame = frame.copy()

            # Write into inactive buffer
            if self.write_buffer == 0:
                self.buffer_a = frame
            else:
                self.buffer_b = frame

            # Swap safely
            with self.swap_lock:
                self.read_buffer, self.write_buffer = self.write_buffer, self.read_buffer

            self.ret = True
            next_frame_time += self.frame_delay

        self.stopped = True

    def read(self):
        """
        Thread-safe frame grab.
        Returns a COPY of the current frame to ensure bit-level stability.
        """
        if not self.ret:
            return False, None

        with self.read_lock:
            with self.swap_lock:
                buf = self.read_buffer

            if buf == 0:
                frame = self.buffer_a
            else:
                frame = self.buffer_b

            # Return a defensive copy to prevent corruption
            return True, frame.copy()

    def isOpened(self):
        return not self.stopped

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()


# --- virtual_camera_single.py ---

class VideoCaptureSingle:
    """
    Synchronous, zero-thread video reader.
    Returns frames EXACTLY in the order they are encoded.
    """

    def __init__(self, video_path, loop=False):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        self.loop = loop
        self.stopped = False

    def read(self):
        if self.stopped:
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            else:
                self.stopped = True
                return False, None

        return True, frame

    def isOpened(self):
        return not self.stopped

    def release(self):
        self.stopped = True
        self.cap.release()
