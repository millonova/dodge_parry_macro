"""Hardware-level low-latency DirectInput keyboard emulation for Windows."""

import ctypes
import queue
import sys
import threading
import time
from typing import Dict, Optional

# Try importing pydirectinput
try:
    import pydirectinput
    pydirectinput.PAUSE = 0.001
    HAS_PYDIRECTINPUT = True
except ImportError:
    HAS_PYDIRECTINPUT = False

PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

SCANCODES: Dict[str, int] = {
    "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
    "6": 0x07, "7": 0x08, "8": 0x09, "9": 0x0A, "0": 0x0B,
    "q": 0x10, "w": 0x11, "e": 0x12, "r": 0x13, "t": 0x14,
    "y": 0x15, "u": 0x16, "i": 0x17, "o": 0x18, "p": 0x19,
    "a": 0x1E, "s": 0x1F, "d": 0x20, "f": 0x21, "g": 0x22,
    "h": 0x23, "j": 0x24, "k": 0x25, "l": 0x26,
    "z": 0x2C, "x": 0x2D, "c": 0x2E, "v": 0x2F, "b": 0x30,
    "n": 0x31, "m": 0x32,
    "space": 0x39, "enter": 0x1C, "return": 0x1C, "tab": 0x0F,
    "shift": 0x2A, "ctrl": 0x1D, "alt": 0x38,
}


class DirectInputHandler:
    """Sends DirectInput scancodes using pydirectinput or Win32 SendInput."""

    def __init__(
        self,
        default_press_duration_ms: float = 40.0,
        pre_press_delay_ms: float = 0.0,
        mock: bool = False,
    ):
        self.default_press_duration = max(0.02, default_press_duration_ms / 1000.0)
        self.pre_press_delay_s = max(0.0, pre_press_delay_ms / 1000.0)
        self.mock = mock or (sys.platform != "win32")
        self.last_pressed_key: Optional[str] = None
        self.total_presses: int = 0
        self.active_keys: set[str] = set()

    def _send_sendinput(self, scan_code: int, flags: int) -> None:
        if self.mock:
            return
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, scan_code, flags, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(INPUT_KEYBOARD), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def key_down(self, key: str) -> None:
        k = key.lower().strip()
        self.active_keys.add(k)
        if self.mock:
            return
        if HAS_PYDIRECTINPUT:
            try:
                pydirectinput.keyDown(k)
                return
            except Exception:
                pass
        code = SCANCODES.get(k)
        if code is not None:
            self._send_sendinput(code, KEYEVENTF_SCANCODE)

    def key_up(self, key: str) -> None:
        k = key.lower().strip()
        self.active_keys.discard(k)
        if self.mock:
            return
        if HAS_PYDIRECTINPUT:
            try:
                pydirectinput.keyUp(k)
                return
            except Exception:
                pass
        code = SCANCODES.get(k)
        if code is not None:
            self._send_sendinput(code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP)

    def release_all_keys(self) -> None:
        """Emergency failsafe: release all held keys and all action scancodes."""
        for k in list(self.active_keys):
            self.key_up(k)
        for k in ["w", "a", "s", "d", "f", "space"]:
            self.key_up(k)
        self.active_keys.clear()

    def press_key_sync(
        self,
        key: str,
        duration_s: Optional[float] = None,
        delay_s: Optional[float] = None,
    ) -> None:
        duration = duration_s if duration_s is not None else self.default_press_duration
        delay = delay_s if delay_s is not None else self.pre_press_delay_s
        if delay > 0:
            time.sleep(delay)
        k = key.lower().strip()
        self.key_down(k)
        time.sleep(duration)
        self.key_up(k)
        self.last_pressed_key = k
        self.total_presses += 1

    def press_key_async(
        self,
        key: str,
        duration_s: Optional[float] = None,
        delay_s: Optional[float] = None,
    ) -> None:
        duration = duration_s if duration_s is not None else self.default_press_duration
        delay = delay_s if delay_s is not None else self.pre_press_delay_s
        k = key.lower().strip()
        self.last_pressed_key = k
        self.total_presses += 1

        def _worker():
            if delay > 0:
                time.sleep(delay)
            self.key_down(k)
            time.sleep(duration)
            self.key_up(k)

        threading.Thread(target=_worker, daemon=True).start()


_audio_queue: queue.Queue = queue.Queue(maxsize=16)
_audio_thread_lock = threading.Lock()
_audio_worker_thread: Optional[threading.Thread] = None


def _audio_worker():
    if sys.platform != "win32":
        return
    try:
        import winsound
    except ImportError:
        return

    while True:
        try:
            cue = _audio_queue.get(timeout=2.0)
        except queue.Empty:
            continue
        try:
            if cue == "trigger":
                winsound.Beep(1200, 35)
            elif cue == "arm":
                winsound.Beep(700, 50)
                winsound.Beep(1100, 60)
            elif cue == "disarm":
                winsound.Beep(1000, 50)
                winsound.Beep(600, 60)
            elif cue == "lock":
                winsound.Beep(1400, 40)
            elif cue == "unlock":
                winsound.Beep(900, 40)
            elif cue == "panic":
                winsound.Beep(450, 120)
                winsound.Beep(320, 150)
        except Exception:
            pass
        finally:
            _audio_queue.task_done()


def play_audio_cue(cue: str, enabled: bool = True) -> None:
    """Non-blocking queued audio feedback using a single persistent daemon worker thread."""
    if not enabled:
        return
    global _audio_worker_thread
    if _audio_worker_thread is None or not _audio_worker_thread.is_alive():
        with _audio_thread_lock:
            if _audio_worker_thread is None or not _audio_worker_thread.is_alive():
                _audio_worker_thread = threading.Thread(target=_audio_worker, daemon=True)
                _audio_worker_thread.start()
    try:
        _audio_queue.put_nowait(cue)
    except queue.Full:
        pass

