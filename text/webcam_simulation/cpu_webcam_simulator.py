# video_process_capture.py
import cv2
import time
import numpy as np
from multiprocessing import Process, Pipe, Value, Event
from multiprocessing import shared_memory
import multiprocessing
import os

try:
    import psutil
except Exception:
    psutil = None


def _set_affinity(core_id):
    """Try set affinity on both Windows and Linux if psutil not available fallback to os.sched_setaffinity."""
    try:
        if psutil:
            psutil.Process(os.getpid()).cpu_affinity([core_id])
        else:
            # Linux fallback; Windows will raise if used without psutil
            try:
                os.sched_setaffinity(0, {core_id})
            except AttributeError:
                # No-op if platform doesn't support
                pass
    except Exception:
        # best-effort; don't fail if affinity can't be set
        pass


def _reader_process(video_path, conn, latest_index, stop_flag, ready_event,
                    loop=False, real_time=False, core=None):
    """
    Child process target that reads frames and writes them into 2 shared-memory buffers.
    After creating buffers it sends metadata via conn to parent:
       {'shm0': name0, 'shm1': name1, 'shape': shape, 'dtype': dtype.str, 'frame_delay': frame_delay}
    """
    if core is not None:
        p = psutil.Process()
        # convert to list if it's a single int
        if isinstance(core, int):
            core_list = [core]
        else:
            core_list = list(core)
        p.cpu_affinity(core_list)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        conn.send({'error': f'Could not open video: {video_path}'})
        conn.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps < 2:
        fps = 30.0
    frame_delay = 1.0 / fps

    # read first frame to determine shape/dtype/size
    ret, frame = cap.read()
    if not ret:
        # video empty
        conn.send({'error': 'Empty video or failed to read first frame.'})
        conn.close()
        cap.release()
        return

    frame = frame.copy()
    shape = frame.shape
    dtype = frame.dtype

    # Create two shared memory blocks sized exactly for one frame each
    nbytes = frame.nbytes
    shm0 = shared_memory.SharedMemory(create=True, size=nbytes)
    shm1 = shared_memory.SharedMemory(create=True, size=nbytes)

    # Map numpy arrays to shms and write initial frame into both
    buf0 = np.ndarray(shape, dtype=dtype, buffer=shm0.buf)
    buf1 = np.ndarray(shape, dtype=dtype, buffer=shm1.buf)
    buf0[:] = frame
    buf1[:] = frame

    # Notify parent about shm names and metadata
    conn.send({'shm0': shm0.name,
               'shm1': shm1.name,
               'shape': shape,
               'dtype': str(dtype),
               'frame_delay': frame_delay})

    # Set latest to 0 meaning buf0 has the newest frame
    with latest_index.get_lock():
        latest_index.value = 0

    # Flip/flop writer index; 1 will be used next
    write_idx = 1
    next_frame_time = time.time() + frame_delay
    ready_event.set()

    try:
        while not stop_flag.value:
            now = time.time()
            delay = next_frame_time - now
            if delay > 0 and real_time:
                time.sleep(delay)
            # read next frame
            ret, frame = cap.read()
            if not ret:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            frame = frame.copy()

            if write_idx == 0:
                buf0[:] = frame
            else:
                buf1[:] = frame

            # publish the index atomically
            with latest_index.get_lock():
                latest_index.value = write_idx

            # rotate writer
            write_idx ^= 1
            next_frame_time += frame_delay
    finally:
        # send stop notice (optional)
        try:
            conn.send({'stopped': True})
        except Exception:
            pass
        # keep objects alive until we cleanup
        conn.close()
        cap.release()
        # child does NOT unlink the shared memory here: parent will handle unlink.
        # Close local shm handles
        shm0.close()
        shm1.close()


class VideoProcessCapture:
    """
    Multiprocessing-based video capture that always provides the latest frame (drops allowed).
    API similar to your previous VideoThreadedCapture:
        cap = VideoProcessCapture(path, loop=False, real_time=False, core=None)
        ret, frame = cap.read()
        cap.release()
    """

    def __init__(self, video_path, loop=False, real_time=False, core=None, startup_timeout=5.0):
        self._is_opened = False
        self.video_path = video_path
        self.loop = loop
        self.real_time = real_time
        self.core = core

        # control/metadata pipe to receive shared-memory names from child
        parent_conn, child_conn = Pipe(duplex=True)
        self._parent_conn = parent_conn

        # atomic integer that indicates which buffer has the latest frame: 0 or 1. -1 => not ready
        self._latest_index = Value('i', -1)  # synchronized value with lock
        self._stop_flag = Value('b', False)  # boolean flag to tell child to stop
        self._ready_event = Event()

        self._proc = Process(target=_reader_process,
                             args=(self.video_path, child_conn,
                                   self._latest_index, self._stop_flag, self._ready_event,
                                   self.loop, self.real_time, self.core),
                             daemon=True)
        self._proc.start()

        # Wait for child to send shm names or an error
        self._shm0 = None
        self._shm1 = None
        self._shape = None
        self._dtype = None
        self._frame_delay = None
        start = time.time()
        try:
            while True:
                if self._parent_conn.poll(0.1):
                    msg = self._parent_conn.recv()
                    if 'error' in msg:
                        self.release()
                        raise RuntimeError(msg['error'])
                    self._shm0_name = msg['shm0']
                    self._shm1_name = msg['shm1']
                    self._shape = tuple(msg['shape'])
                    self._dtype = np.dtype(msg['dtype'])
                    self._frame_delay = msg.get('frame_delay', None)
                    # open and map shared memory
                    self._shm0 = shared_memory.SharedMemory(name=self._shm0_name)
                    self._shm1 = shared_memory.SharedMemory(name=self._shm1_name)
                    self._buf0 = np.ndarray(self._shape, dtype=self._dtype, buffer=self._shm0.buf)
                    self._buf1 = np.ndarray(self._shape, dtype=self._dtype, buffer=self._shm1.buf)
                    break
                if time.time() - start > startup_timeout:
                    self.release()
                    raise TimeoutError("Timed out waiting for webcam process to create shared memory.")
        except Exception:
            # clean up if we failed during init
            self.release()
            raise

        self._is_opened = True

    def read(self):
        """Return (ret, frame). Frame is a copy of the latest shared buffer (safe for caller)."""
        if not self._is_opened:
            return False, None

        idx = None
        # read the index under lock to get a consistent value
        with self._latest_index.get_lock():
            idx = self._latest_index.value

        if idx == -1:
            return False, None

        if idx == 0:
            # copy the frame out (safe)
            return True, self._buf0.copy()
        else:
            return True, self._buf1.copy()

    def isOpened(self):
        return self._is_opened and (not self._stop_flag.value)

    def release(self, wait=True, timeout=2.0):
        """Stop the child process and cleanup shared memory."""
        if not self._is_opened:
            return

        # ask child to stop
        self._stop_flag.value = True

        # wait for child process to exit
        if wait and self._proc.is_alive():
            self._proc.join(timeout)
            if self._proc.is_alive():
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                self._proc.join(0.5)

        # Close and unlink shared memory (parent-owned cleanup)
        try:
            if hasattr(self, '_shm0') and self._shm0 is not None:
                self._shm0.close()
                self._shm0.unlink()
        except Exception:
            pass
        try:
            if hasattr(self, '_shm1') and self._shm1 is not None:
                self._shm1.close()
                self._shm1.unlink()
        except Exception:
            pass

        self._is_opened = False

    # convenience context manager
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
