"""
A videoCapture API extension that allows for subsampling and threading
"""
from collections import deque
from pathlib import Path
from typing import Union, Literal, Tuple
from time import time
import threading
import warnings
import tqdm

import numpy as np
import cv2


class VideoCapture:
    """
    extends [cv2::VideoCapture class](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html)
    for video or stream subsampling.

    Parameters
    ----------
    filename : Union[str, int]
        Open video file or image file sequence or a capturing device
        or a IP video stream for video capturing.
    target_fps : int, optional
        the target frame rate. To ensure a constant time period between
        each subsampled frames, this parameter is used to compute a
        stride for the capture reading. For instance,
        if the original capture is 64fps and you want a 30fps capture
        out, it is going to take one frame over two giving an effective
        frame rate of 32fps.
        If None, will extract every frame of the capture.
    buffer_size : int, optional
        the size of the reading buffer.
    capture_fps : float, optional
        the input frame rate. Can be specified to avoid the frame rate
        estimation part as it can be imprecise or couldn't work if the
        stream does not emit any frame for now.
    drop_warning_enable : bool, optional
        whether or not to show a warning if a frame had to be dropped.
        When the buffer is full and a new frame is coming in, a frame
        has to be dropped in order to make space in the buffer.
        By default True.
    tqdm_progressbar : tqdm.tqdm, optional
        progress bar to display
    """
    def __init__(self,
                 filename: Union[Path, str, int],
                 *,
                 use_threading: bool = False,
                 target_fps: int = None,
                 stride: int = None,
                 buffer_size: int = 5,
                 capture_fps: float = None,
                 drop_warning_enable: bool = True,
                 tqdm_progressbar: tqdm.tqdm = None,
                 ) -> None:
        _filename = filename
        if isinstance(filename, Path):
            _filename = str(filename)
        self._cap = cv2.VideoCapture(_filename)
        self.cap_name = filename
        self.mode = self.get_cap_mode(filename)

        self.frame_count = 0
        self._capture_fps = self.get_capture_fps(capture_fps)
        self.stride = self.get_stride(target_fps, stride)
        self._out_fps = self._capture_fps / self.stride

        self.use_threading = use_threading
        if use_threading:
            self._imgs_queue = Queue(maxlen=buffer_size)
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._thread_read)
            self._thread.start()

        self.tqdm = None
        if isinstance(tqdm_progressbar, tqdm.tqdm):
            self.tqdm = tqdm_progressbar
            self.tqdm.total = int(int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) + 1) / self.stride)

        self.drop_warning_enable = drop_warning_enable

    @property
    def current_frame_index(self) -> int:
        """return a frame id that is the index of the frame * the
        subsample rate"""
        return self.frame_count

    def get_capture_fps(self, value: Union[None, float]) -> float:
        """
        return the input capture frame rate, either from its property
        or via an estimation, or the user input if provided.

        Parameters
        ----------
        value : Union[None, float]
            user input

        Returns
        -------
        float
            the capture frame rate.
        """
        if value is None:
            # cv2.CAP_PROP_FPS returns 0 if the property doesn't exist
            capture_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if capture_fps == 0:
                print("aya")
                capture_fps = self.estimate_capture_fps()
        else:
            capture_fps = value

        return capture_fps

    def estimate_capture_fps(self, evaluation_period: int = 5):
        """evaluate the frame rate over a period of 5 seconds"""
        frame_count = 0
        while self.isOpened():
            ret, _ = self._cap.read()
            if ret is True:
                if frame_count == 0:
                    start = time()

                frame_count += 1

                if time() - start > evaluation_period:
                    break

        if frame_count == 0:
            raise FileNotFoundError("Could not estimate the input capture fps. You can specify "
                                    "the input frame rate using the `capture_fps` argument")

        return round(frame_count / (time() - start), 2)

    def get_stride(self, target_fps, stride):
        """compute the stride of the reading process"""
        if target_fps is not None and stride is not None:
            raise ValueError("Both `target_fps` and `stride` are set. "
                             "Only one can be used at a time.")

        if target_fps is None and stride is None:
            return 1

        if target_fps is not None:
            stride = round(self._capture_fps / target_fps)

            if stride == 0:
                raise ValueError("target_fps is higher than the capture frame rate")

        return stride

    def get_cap_mode(self, filename: Union[Path, str, int]) -> Literal["stream", "video"]:
        """
        tries to identify what is the provided file.

        Parameters
        ----------
        filename : Union[Path, str, int]

        Returns
        -------
        str
            name of the capture mode, either "stream" or "video".
        """
        if isinstance(filename, (int, float)):
            return "stream"

        filename = Path(filename)
        if filename.is_file():
            return "video"

        return "stream"

    def isOpened(self):
        """Returns true if video capturing has been initialized already."""
        return self._cap.isOpened()

    def release(self):
        """Closes video file or capturing device."""
        self._cap.release()
        if self.tqdm is not None:
            self.tqdm.close()
        if self.use_threading:
            self._stop_event.set()
            self._thread.join()

    def read(self) -> Tuple[bool, np.ndarray]:
        """
        Grabs, decodes and returns the next subsampled video frame.
        If there is no image in the queue, wait for one to arrive.
        """
        if self.mode == "stream" and self.use_threading:
            while self._imgs_queue.is_empty():
                if self._stop_event.is_set():
                    return None

                continue

            self.tqdm.update()
            return self._imgs_queue.popleft()
        else:
            _, frame = self._cap.read()

            self.frame_count += 1
            if self.frame_count % self.stride != 0:
                return self.read()

            self.tqdm.update()
            return self.frame_count, frame

    def _thread_read(self):
        """
        read frame from a input capture and put them in a buffer.
        """
        self.frame_count = 0
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()

            if ret is False:
                continue

            self.frame_count += 1
            if self.frame_count % self.stride != 0:
                continue

            if self.mode == "stream":
                # this avoid the queue shrinking before the result of
                # full() can be used.
                with self._imgs_queue.lock:
                    if self._imgs_queue.maxlen <= len(self._imgs_queue):
                        self._imgs_queue.popleft()

                        if self.drop_warning_enable:
                            warnings.warn("frame dropped")
            if self.mode == "video":
                # when dealing with a video, we can wait for the queue
                # being not full
                while self._imgs_queue.is_full():
                    if self._stop_event.is_set():
                        return None
                    continue

            # we don't need the lock as this is the only thread that
            # can put things in the queue
            self._imgs_queue.append((self.frame_count, frame))

    def __del__(self):
        if self.isOpened():
            self.release()


class Queue(deque):
    """the queue.Queue often used for threading was too high level
    and did not allow enough control over the lock. as a result,
    "private" methods were too often used so this class is a lighter
    queue.Queue implementation better suited for our use case."""
    def __init__(self, maxlen: int):
        self.lock = threading.Lock()
        super().__init__(maxlen=maxlen)

    def is_empty(self):
        """check if the queue is empty"""
        return len(self) == 0

    def is_full(self):
        """check if the queue is full"""
        return len(self) == self.maxlen