# decoding_pipeline/pipeline.py

import multiprocessing
import psutil
from decoding_pipeline.shared_functions import shared_class
from decoding_pipeline.decoder_worker import decoding_worker
from decoding_pipeline.message_worker import message_worker
from decoding_pipeline.audio_worker import audio_worker
from decoding_pipeline.watchdog import watchdog

# Shared objects
_frame_queue = None
_command_queue = None
_bitgrid_queue = None
_message_queue = None
_audio_queue = None

_stop_flag = None
_last_frame = None
_recall_flag = None

_last_decode_timestamp = None
_last_message_timestamp = None


# --- Pipeline ---

class pipeline_message:

    def __init__(self):
        # Processes
        self._decode_process = None
        self._message_process = None
        self._watchdog_process = None


    def start_pipeline(self, core_decode_worker=None, core_message_worker=None, core_watchdog=None, queue_maxsize=100):
        """
        Starts the decoding worker and watchdog processes.

        Arguments:
            core_decode_worker (list[int] | None): CPU cores for decoding worker.
            core_watchdog (list[int] | None): CPU cores for watchdog.
            queue_maxsize (int): Maximum size of the frame queue.
        """

        global _frame_queue, _command_queue, _bitgrid_queue, \
            _message_queue, _recall_flag, _last_frame, _stop_flag, \
            _last_decode_timestamp, _last_message_timestamp
               

        # Shared objects
        shared_class.initialize_shared_objects(queue_maxsize)

        (
            self._frame_queue, self._command_queue, self._bitgrid_queue,
            self._message_queue, _, 
            self._stop_flag, self._last_frame, self._recall_flag,
            self._last_decode_timestamp, self._last_message_timestamp
        ) = shared_class.get_shared_objects()

        # Start decoding worker process
        self._decode_process = multiprocessing.Process(
            target=decoding_worker,
            args=(_frame_queue, _command_queue, _bitgrid_queue, 
                  _stop_flag, _last_frame, _last_decode_timestamp),
            daemon=True
        )
        self._decode_process.start()

        # Pin decoding worker to specific cores
        if core_decode_worker:
            try:
                if isinstance(core_decode_worker, int):
                    core_decode_worker = [core_decode_worker]  # Convert single core to list
                else:
                    core_decode_worker = list(core_decode_worker)  # Ensure it's a list
                psutil.Process(self._decode_process.pid).cpu_affinity(core_decode_worker)
            except Exception as e:
                print(f"[WARNING] Could not pin decoding worker cores: {e}")

        # Start message worker process
        self._message_process = multiprocessing.Process(
            target=message_worker,
            args=(_bitgrid_queue, _message_queue, _recall_flag, 
                  _last_frame, _stop_flag, _last_message_timestamp),
            daemon=True
        )
        self._message_process.start()

        # Pin message worker to specific cores
        if core_message_worker:
            try:
                if isinstance(core_message_worker, int):
                    core_message_worker = [core_message_worker]
                else:
                    core_message_worker = list(core_message_worker)
                psutil.Process(self._message_process.pid).cpu_affinity(core_message_worker)
            except Exception as e:
                print(f"[WARNING] could not pin message worker cores: {e}")

        # Start watchdog process
        self._watchdog_process = multiprocessing.Process(
            target=watchdog,
            args=(_last_decode_timestamp, _stop_flag),
            daemon=True
        )
        self._watchdog_process.start()

        # Pin watchdog to specific cores
        if core_watchdog:
            try:
                if isinstance(core_watchdog, int):
                    core_watchdog = [core_watchdog]  # Convert single core to list
                else:
                    core_watchdog = list(core_watchdog)  # Ensure it's a list
                psutil.Process(self._watchdog_process.pid).cpu_affinity(core_watchdog)
            except Exception as e:
                print(f"[WARNING] Could not pin watchdog cores: {e}")


    def stop_pipeline(self):

        global _frame_queue, _command_queue, _bitgrid_queue, \
               _message_queue, _stop_flag

        if _stop_flag is None:
            return

        # signal shutdown
        _stop_flag.value = True
        # send explicit shutdown command
        try:
            _command_queue.put_nowait(("shutdown", None))
        except:
            pass

        # send sentinels to queues to wake blocked gets
        try: _frame_queue.put_nowait(None)
        except: pass
        try: _bitgrid_queue.put_nowait(None)
        except: pass

        # join processes only if started and alive
        for proc, name in ((self._decode_process,"decode"), (self._message_process,"message"), (self._watchdog_process,"watchdog")):
            if proc is None:
                print(f"[Pipeline] {name} process object is None — skipping join")
                continue
            if getattr(proc, "_popen", None) is None:
                print(f"[Pipeline] {name} process never started — skipping join")
                continue
            if proc.is_alive():
                proc.join(timeout=3)
                if proc.is_alive():
                    print(f"[Pipeline] {name} still alive — terminating")
                    proc.terminate()
                    proc.join(timeout=1)


class pipeline_audio:

    def __init__(self):
        # Processes
        self._decode_process = None
        self._audio_process = None
        self._watchdog_process = None


    def start_pipeline(self, core_decode_worker=None, core_audio_worker=None, core_watchdog=None, queue_maxsize=100):
        """
        Starts the decoding worker and watchdog processes.

        Arguments:
            core_decode_worker (list[int] | None): CPU cores for decoding worker.
            core_audio_worker (list[int] | None): CPU cores for audio worker.
            core_watchdog (list[int] | None): CPU cores for watchdog.
            queue_maxsize (int): Maximum size of the frame queue.
        """

        global _frame_queue, _command_queue, _bitgrid_queue, \
            _audio_queue, _recall_flag, _last_frame, _stop_flag, \
            _last_decode_timestamp, _last_message_timestamp
               

        # Shared objects
        shared_class.initialize_shared_objects(queue_maxsize)

        (
            self._frame_queue, self._command_queue, self._bitgrid_queue,
            _, self._audio_queue, 
            self._stop_flag, self._last_frame, self._recall_flag,
            self._last_decode_timestamp, self._last_message_timestamp
        ) = shared_class.get_shared_objects()

        # Start decoding worker process
        self._decode_process = multiprocessing.Process(
            target=decoding_worker,
            args=(_frame_queue, _command_queue, _bitgrid_queue, 
                  _stop_flag, _last_frame, _last_decode_timestamp),
            daemon=True
        )
        self._decode_process.start()

        # Pin decoding worker to specific cores
        if core_decode_worker:
            try:
                if isinstance(core_decode_worker, int):
                    core_decode_worker = [core_decode_worker]  # Convert single core to list
                else:
                    core_decode_worker = list(core_decode_worker)  # Ensure it's a list
                psutil.Process(self._decode_process.pid).cpu_affinity(core_decode_worker)
            except Exception as e:
                print(f"[WARNING] Could not pin decoding worker cores: {e}")

        # Start audio worker process
        self._audio_process = multiprocessing.Process(
            target=audio_worker,
            args=(_bitgrid_queue, _audio_queue, 
                _stop_flag, _last_message_timestamp),
            daemon=True
        )
        self._audio_process.start()

        # Pin audio worker to specific cores
        if core_audio_worker:
            try:
                if isinstance(core_audio_worker, int):
                    core_audio_worker = [core_audio_worker]
                else:
                    core_audio_worker = list(core_audio_worker)
                psutil.Process(self._audio_process.pid).cpu_affinity(core_audio_worker)
            except Exception as e:
                print(f"[WARNING] could not pin message worker cores: {e}")

        # Start watchdog process
        self._watchdog_process = multiprocessing.Process(
            target=watchdog,
            args=(_last_decode_timestamp, _stop_flag),
            daemon=True
        )
        self._watchdog_process.start()

        # Pin watchdog to specific cores
        if core_watchdog:
            try:
                if isinstance(core_watchdog, int):
                    core_watchdog = [core_watchdog]  # Convert single core to list
                else:
                    core_watchdog = list(core_watchdog)  # Ensure it's a list
                psutil.Process(self._watchdog_process.pid).cpu_affinity(core_watchdog)
            except Exception as e:
                print(f"[WARNING] Could not pin watchdog cores: {e}")


    def stop_pipeline(self):

        global _frame_queue, _command_queue, _bitgrid_queue, \
               _message_queue, _stop_flag

        if _stop_flag is None:
            return

        # signal shutdown
        _stop_flag.value = True
        # send explicit shutdown command
        try:
            _command_queue.put_nowait(("shutdown", None))
        except:
            pass

        # send sentinels to queues to wake blocked gets
        try: _frame_queue.put_nowait(None)
        except: pass
        try: _bitgrid_queue.put_nowait(None)
        except: pass

        # join processes only if started and alive
        for proc, name in ((self._decode_process,"decode"), (self._audio_process,"audio"), (self._watchdog_process,"watchdog")):
            if proc is None:
                print(f"[Pipeline] {name} process object is None — skipping join")
                continue
            if getattr(proc, "_popen", None) is None:
                print(f"[Pipeline] {name} process never started — skipping join")
                continue
            if proc.is_alive():
                proc.join(timeout=3)
                if proc.is_alive():
                    print(f"[Pipeline] {name} still alive — terminating")
                    proc.terminate()
                    proc.join(timeout=1)


pip_message = pipeline_message()
pip_audio = pipeline_audio()

