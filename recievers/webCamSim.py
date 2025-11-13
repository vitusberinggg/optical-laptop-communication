# --- virtual_camera.py ---

import cv2
import threading
import time

class VideoThreadedCapture:
    """
    Threaded video capture to simulate a webcam from a video file.
    """

    def __init__(self, video_path, loop=False):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.loop = loop
        self.latest_frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()

        # Get FPS info
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30
        self.frame_delay = 1.0 / self.fps

        # Start the thread
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        """
        Continuously read frames in a separate thread.
        """
        while not self.stopped:
            ret, frame = self.cap.read()

            if not ret:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.stopped = True
                    break

            with self.lock:
                self.latest_frame = frame.copy()
                self.ret = ret

            time.sleep(self.frame_delay)  # simulate real-time capture

    def read(self):
        """
        Returns the latest frame.
        """
        with self.lock:
            return self.ret, self.latest_frame.copy() if self.latest_frame is not None else None

    def isOpened(self):
        return not self.stopped

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()
