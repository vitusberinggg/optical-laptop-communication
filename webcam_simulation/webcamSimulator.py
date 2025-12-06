
# --- Imports ---

import cv2
import threading
import time

from distorter import FrameDistorter

# --- Definitions ---

sender_output_width = 1920 # Width of the sender output in pixels
sender_output_height = 1200 # Height of the sender output in pixels

# --- Main class ---

class VideoThreadedCapture:

    """
    Bit-precise, frame-accurate simulator for webcam-style video:

    - No race conditions
    - No partial frame overwrites
    - Correct FPS timing
    - Real-time (drop frames) or exact playback (no drops)

    """

    def __init__(self, video_path, loop = False, real_time = False):

        """

        Arguments:
            "self"
            "video_path"
            "loop"
            "real_time"

        Returns:
            None
        
        """

        self.cap = cv2.VideoCapture(video_path)

        self.distorter = FrameDistorter(preset="heavy")

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.loop = loop
        self.real_time = real_time

        self.buffer_a = None
        self.buffer_b = None

        self.read_buffer = 0
        self.write_buffer = 1

        self.swap_lock = threading.Lock()
        self.read_lock = threading.Lock()

        self.ret = False
        self.stopped = False

        self.now1 = time.time()

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        if fps < 2:
            fps = 30

        self.frame_delay = 1.0 / fps

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):

        """
        Background frame reader.

        Arguments:
            "self"

        Returns:
            None
        
        """

        next_frame_time = time.time()

        while not self.stopped:

            now = time.time()
            delay = next_frame_time - now

            if delay > 0:
                time.sleep(delay)

            if time.time() - self.now1 > 0.5:
                print(f"Delay is: {delay}")
                self.now1 = time.time()

            ret, frame = self.cap.read()

            if not ret:

                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                break

            frame = frame.copy()

            frame = self.distorter.apply(frame)

            if self.write_buffer == 0:
                self.buffer_a = frame

            else:
                self.buffer_b = frame

            with self.swap_lock:
                self.read_buffer, self.write_buffer = self.write_buffer, self.read_buffer

            self.ret = True
            next_frame_time += self.frame_delay

        self.stopped = True

    def read(self):

        """
        Thread-safe frame grab. Returns a COPY of the current frame to ensure bit-level stability.

        Arguments:
            "self"
        
        Returns:
            True/False
            "frame.copy"
        
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

            return True, frame.copy()

    def isOpened(self):

        """

        """

        return not self.stopped

    def release(self):

        """
        
        """
        
        self.stopped = True

        if self.thread.is_alive():
            self.thread.join()

        self.cap.release()

class VideoCaptureSingle:

    """
    Synchronous, zero-thread video reader. Returns frames EXACTLY in the order they are encoded.

    """

    def __init__(self, video_path, loop = False):

        """
        
        """

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.loop = loop
        self.stopped = False

    def read(self):

        """
        
        """

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

        """

        """

        return not self.stopped

    def release(self):

        """

        """

        self.stopped = True
        self.cap.release()

# --- Testing Web Cam Simulator ---

if __name__ == "__main__":



    video_path = r"C:\Users\ejadmax\code\optical-laptop-communication\webcam_simulation\sender_v5.mp4"
    video_capture = VideoThreadedCapture(video_path)

    if not video_capture.isOpened():
        print("\n[WARNING] Couldn't start video capture.")
        exit()

    while True:

        read_was_sucessful, frame = video_capture.read() # Tries to grab one initial frame to make sure the video capture is "warmed up"

        if read_was_sucessful:
            break

        time.sleep(0.01)

    cv2.namedWindow("sjöbo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("sjöbo", sender_output_width, sender_output_height)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            cv2.imshow("sjöbo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video_capture.release()
    cv2.destroyAllWindows()