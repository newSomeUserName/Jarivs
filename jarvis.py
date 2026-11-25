#!/usr/bin/env python
"""
J.A.R.V.I.S. Hand Control — управление компьютером жестами рук через веб-камеру (Windows).

Левая рука   = мышь (курсор, клик, перетаскивание, правый клик, прокрутка).
Правая рука  = команды (громкость, свайпы, медиа, скриншот).
Подробности — в README.md. Настройки — в config.json (создаётся при первом запуске).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
import urllib.request
from collections import deque
from ctypes import wintypes as wt
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
CONFIG_PATH = APP_DIR / "config.json"
WINDOW = "JARVIS Hand Control"

DEFAULT_CONFIG = {
    "camera_index": 0,
    "camera_width": 640,
    "camera_height": 480,
    "mirror": True,
    "primary_hand": "Left",
    # Режим курсора: "relative" — как тачпад (курсор сдвигается на столько, на сколько сдвинулась рука,
    # с ускорением); "absolute" — положение руки в рамке = положение курсора на экране.
    "cursor_mode": "relative",
    # Чувствительность (меняется клавишами F7/F8 прямо в программе).
    "sensitivity": 1.0,
    # Кривая скорости в режиме тачпада (как «повышенная точность» мыши в Windows).
    # Медленно (скорость руки < speed_slow) -> усиление gain_slow (точно).
    # Быстро (> speed_fast) -> усиление gain_fast (далеко). Скорость — в ширинах кадра в секунду.
    "gain_slow": 0.8,
    "gain_fast": 5.0,
    "speed_slow": 0.1,
    "speed_fast": 0.8,
    # Движения руки меньше этого (доля кадра) считаются дрожанием и игнорируются.
    "deadzone": 0.0015,
    # Прицеливание: когда большой и указательный почти сведены, курсор замедляется.
    "precision_start": 0.65,
    "precision_gain": 0.35,
    # Зона кадра (x0, y0, x1, y1 в долях 0..1) для режима absolute при чувствительности 1.
    "active_region": [0.22, 0.18, 0.78, 0.68],
    # Сглаживание курсора (One Euro filter): меньше min_cutoff -> плавнее, больше beta -> отзывчивее.
    "smooth_min_cutoff": 0.7,
    "smooth_beta": 4.0,
    "catchup": 0.8,
    # Щипки (расстояние между кончиками пальцев / размер ладони).
    "pinch_on": 0.28,
    "pinch_off": 0.45,
    "pinch_min_reach": 0.55,
    # Кулак / открытая ладонь (расстояние кончиков пальцев от центра ладони / размер ладони).
    "fist_curl": 0.7,
    "open_extend": 0.9,
    "fist_frames": 4,
    # Кулак срабатывает, только если перед ним была открытая ладонь (не раньше чем N секунд назад).
    "fist_arm_time": 1.5,
    "drag_threshold": 0.02,
    # Сколько секунд держать щипок неподвижно для правого клика.
    "long_press": 0.6,
    "release_freeze": 0.15,
    "pause_hold": 0.8,
    "scroll_speed": 3000,
    "natural_scroll": True,
    "volume_step": 0.035,
    "swipe_distance": 0.22,
    "swipe_time": 0.35,
    "swipe_cooldown": 1.0,
    "preview_width": 560,
    "always_on_top": True,
    # Действия левой руки: сочетания клавиш через "+", например "ctrl+win+right".
    "actions": {
        "swipe_left": "ctrl+win+left",
        "swipe_right": "ctrl+win+right",
        "swipe_up": "win+tab",
        "swipe_down": "win+d",
        "fist": "play_pause",
        "middle_pinch": "win+shift+s",
    },
}


# --------------------------------------------------------------------------- config

def load_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if CONFIG_PATH.exists():
        try:
            user = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            actions = user.pop("actions", {})
            cfg.update(user)
            cfg["actions"].update(actions)
        except (OSError, ValueError) as e:
            print(f"[!] Не удалось прочитать config.json ({e}), использую настройки по умолчанию.")
    else:
        CONFIG_PATH.write_text("{}\n", encoding="utf-8")
    return cfg


def save_config(cfg: dict) -> None:
    """Сохраняет только то, что отличается от настроек по умолчанию."""
    diff = {k: v for k, v in cfg.items() if k != "actions" and DEFAULT_CONFIG.get(k) != v}
    if cfg["actions"] != DEFAULT_CONFIG["actions"]:
        diff["actions"] = cfg["actions"]
    try:
        CONFIG_PATH.write_text(json.dumps(diff, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        print(f"[!] Не удалось сохранить config.json: {e}")


def ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Скачиваю модель распознавания рук MediaPipe (~8 МБ)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# --------------------------------------------------------------------------- Windows input

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetForegroundWindow.restype = wt.HWND
user32.FindWindowW.restype = wt.HWND
user32.FindWindowW.argtypes = (wt.LPCWSTR, wt.LPCWSTR)

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wt.DWORD), ("wParamL", wt.WORD), ("wParamH", wt.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _INPUTUNION)]


VK = {
    "ctrl": 0xA2, "alt": 0xA4, "shift": 0xA0, "win": 0x5B,
    "tab": 0x09, "esc": 0x1B, "enter": 0x0D, "space": 0x20, "backspace": 0x08,
    "delete": 0x2E, "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28, "printscreen": 0x2C,
    "volume_mute": 0xAD, "volume_down": 0xAE, "volume_up": 0xAF,
    "next_track": 0xB0, "prev_track": 0xB1, "play_pause": 0xB3,
}
VK.update({chr(c).lower(): c for c in range(ord("A"), ord("Z") + 1)})
VK.update({str(d): 0x30 + d for d in range(10)})
VK.update({f"f{i}": 0x6F + i for i in range(1, 13)})
EXTENDED_VK = {0x5B, 0x25, 0x26, 0x27, 0x28, 0x2E, 0x24, 0x23, 0x21, 0x22, 0x2C,
               0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB3}


def get_cursor_pos() -> np.ndarray:
    pt = wt.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return np.array([float(pt.x), float(pt.y)])


def set_dpi_aware() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


class InputInjector:
    """Отправляет события мыши и клавиатуры через WinAPI SendInput."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.vx = user32.GetSystemMetrics(76)
        self.vy = user32.GetSystemMetrics(77)
        self.vw = user32.GetSystemMetrics(78)
        self.vh = user32.GetSystemMetrics(79)

    def _send(self, *inputs: INPUT) -> None:
        if self.dry_run:
            return
        arr = (INPUT * len(inputs))(*inputs)
        user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))

    def _mouse(self, flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> INPUT:
        return INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(dx, dy, data & 0xFFFFFFFF, flags, 0, 0))

    def move_to(self, x: float, y: float) -> None:
        nx = int(round((x - self.vx) * 65535 / max(self.vw - 1, 1)))
        ny = int(round((y - self.vy) * 65535 / max(self.vh - 1, 1)))
        self._send(self._mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, nx, ny))

    def left_down(self) -> None:
        self._log("LMB down")
        self._send(self._mouse(MOUSEEVENTF_LEFTDOWN))

    def left_up(self) -> None:
        self._log("LMB up")
        self._send(self._mouse(MOUSEEVENTF_LEFTUP))

    def right_click(self) -> None:
        self._log("RMB click")
        self._send(self._mouse(MOUSEEVENTF_RIGHTDOWN), self._mouse(MOUSEEVENTF_RIGHTUP))

    def wheel(self, amount: int, horizontal: bool = False) -> None:
        self._send(self._mouse(MOUSEEVENTF_HWHEEL if horizontal else MOUSEEVENTF_WHEEL, data=amount))

    def hotkey(self, combo: str) -> bool:
        keys = []
        for name in combo.lower().replace(" ", "").split("+"):
            if name not in VK:
                print(f"[!] Неизвестная клавиша '{name}' в '{combo}'")
                return False
            keys.append(VK[name])
        self._log(f"keys {combo}")

        def key(vk: int, up: bool) -> INPUT:
            flags = (KEYEVENTF_KEYUP if up else 0) | (KEYEVENTF_EXTENDEDKEY if vk in EXTENDED_VK else 0)
            return INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(vk, 0, flags, 0, 0))

        self._send(*[key(k, False) for k in keys], *[key(k, True) for k in reversed(keys)])
        return True

    def _log(self, msg: str) -> None:
        if self.dry_run:
            print(f"  [dry-run] {msg}")


# --------------------------------------------------------------------------- hand model

def _d(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


PALM_IDX = [0, 5, 9, 13, 17]
FINGER_TIPS = (8, 12, 16, 20)


class Hand:
    def __init__(self, lms, w: int, h: int, label: str, score: float):
        self.label = label
        self.score = score
        self.aspect = h / w
        self.norm = np.array([[p.x, p.y] for p in lms], dtype=np.float64)
        self.px = self.norm * np.array([w, h])
        self.scale = max(_d(self.px[0], self.px[9]), 1.3 * _d(self.px[5], self.px[17]), 1e-3)
        self.palm_px = self.px[PALM_IDX].mean(axis=0)
        self.palm_norm = self.norm[PALM_IDX].mean(axis=0)

    def pinch(self, tip: int) -> float:
        return _d(self.px[4], self.px[tip]) / self.scale

    def reach(self, tip: int) -> float:
        return _d(self.px[tip], self.palm_px) / self.scale

    def is_fist(self, cfg) -> bool:
        return all(self.reach(t) < cfg["fist_curl"] for t in FINGER_TIPS)

    def is_open(self, cfg) -> bool:
        return all(self.reach(t) > cfg["open_extend"] for t in FINGER_TIPS)


def parse_hands(result, w: int, h: int, mirror: bool) -> list[Hand]:
    hands = []
    for lms, handed in zip(result.hand_landmarks, result.handedness):
        label, score = ("Right", 0.0)
        if handed:
            label, score = handed[0].category_name, handed[0].score
        # MediaPipe считает, что кадр зеркальный (селфи). Без зеркала — меняем метки.
        if not mirror:
            label = {"Left": "Right", "Right": "Left"}.get(label, label)
        hands.append(Hand(lms, w, h, label, score))
    if len(hands) == 2 and hands[0].label == hands[1].label:
        hands.sort(key=lambda hd: hd.palm_norm[0])
        hands[0].label, hands[1].label = ("Left", "Right") if mirror else ("Right", "Left")
    return hands


class OneEuro:
    """One Euro filter: убирает дрожание в покое, не добавляя задержки при быстрых движениях."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class PinchTracker:
    """Отслеживает щипки большого пальца с указанными пальцами (с гистерезисом)."""

    def __init__(self, tips: tuple[int, ...]):
        self.tips = tips
        self.active: int | None = None
        self.since = 0.0

    def reset(self) -> None:
        self.active = None

    def update(self, hand: Hand, cfg, now: float):
        started = ended = None
        if self.active is not None:
            if hand.pinch(self.active) > cfg["pinch_off"]:
                ended, self.active = self.active, None
        elif self.tips:
            ds = sorted((hand.pinch(t), t) for t in self.tips)
            d0, t0 = ds[0]
            d1 = ds[1][0] if len(ds) > 1 else 9.0
            reach_ok = t0 == 20 or hand.reach(t0) > cfg["pinch_min_reach"]
            if d0 < cfg["pinch_on"] and d1 - d0 > 0.05 and reach_ok:
                self.active, self.since, started = t0, now, t0
        return started, ended


# --------------------------------------------------------------------------- controllers

class MouseController:
    """Основная рука: курсор, клики, перетаскивание, прокрутка, пауза."""

    def __init__(self, cfg, inp: InputInjector, toast):
        self.cfg, self.inp, self.toast = cfg, inp, toast
        self.filter = OneEuro(cfg["smooth_min_cutoff"], cfg["smooth_beta"])
        self.pinches = PinchTracker((8, 12, 20))
        self.mode = "searching"
        self.paused = False
        self.left_down = False
        self.pinch_anchor = None
        self.drag_unlocked = False
        self.pinch_consumed = False
        self.freeze_until = 0.0
        self.pause_fired = False
        self.fist_frames = 0
        self.scroll_last = None
        self.scroll_acc = np.zeros(2)
        self.cursor = None
        self.offset = np.zeros(2)
        self.was_moving = False
        self.engaged = False
        self.last_open = -99.0
        self.rel_prev = None
        self.rel_t = 0.0
        self.precision = False

    # -- helpers
    def _release(self) -> None:
        if self.left_down:
            self.inp.left_up()
            self.left_down = False

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self._release()
        self.toast("Пауза" if self.paused else "Управление включено")

    def region(self) -> list[float]:
        """Рамка для режима absolute: сжимается вокруг центра с ростом чувствительности."""
        x0, y0, x1, y1 = self.cfg["active_region"]
        k = 1.0 / max(self.cfg["sensitivity"], 0.1)
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 * k, (y1 - y0) / 2 * k
        return [cx - hw, cy - hh, cx + hw, cy + hh]

    def _map(self, p: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.region()
        u = min(max((p[0] - x0) / (x1 - x0), 0.0), 1.0)
        v = min(max((p[1] - y0) / (y1 - y0), 0.0), 1.0)
        return np.array([self.inp.vx + u * (self.inp.vw - 1), self.inp.vy + v * (self.inp.vh - 1)])

    def _follow(self, smooth: np.ndarray, now: float, moving: bool, hand: Hand) -> None:
        if self.cfg["cursor_mode"] == "relative":
            self._follow_relative(smooth, now, moving, hand)
        else:
            self._follow_absolute(self._map(smooth), moving)

    def _gain(self, speed: float, hand: Hand) -> float:
        cfg = self.cfg
        x = (speed - cfg["speed_slow"]) / max(cfg["speed_fast"] - cfg["speed_slow"], 1e-6)
        x = min(max(x, 0.0), 1.0)
        x = x * x * (3 - 2 * x)  # плавный переход (smoothstep)
        gain = (cfg["gain_slow"] + (cfg["gain_fast"] - cfg["gain_slow"]) * x) * cfg["sensitivity"]
        # Прицеливание перед кликом: пальцы почти сведены -> медленнее.
        p0, p1 = cfg["pinch_on"], cfg["precision_start"]
        pinch = hand.pinch(8)
        if pinch < p1:
            k = min(max((pinch - p0) / max(p1 - p0, 1e-6), 0.0), 1.0)
            gain *= cfg["precision_gain"] + (1 - cfg["precision_gain"]) * k
            self.precision = k < 0.999
        else:
            self.precision = False
        return gain

    def _follow_relative(self, smooth: np.ndarray, now: float, moving: bool, hand: Hand) -> None:
        if not moving:
            self.rel_prev, self.rel_t = smooth.copy(), now
            self.was_moving = False
            self.precision = False
            return
        if not self.was_moving or self.rel_prev is None:
            self.cursor = get_cursor_pos()  # продолжаем с того места, где курсор сейчас
            self.rel_prev, self.rel_t = smooth.copy(), now
            self.was_moving = True
            return
        d = (smooth - self.rel_prev) * np.array([1.0, hand.aspect])
        dist = float(np.linalg.norm(d))
        if dist < self.cfg["deadzone"]:
            return  # дрожание: копим, пока движение не станет заметным
        speed = dist / max(now - self.rel_t, 1e-3)
        step = d * user32.GetSystemMetrics(0) * self._gain(speed, hand)
        inp = self.inp
        pos = self.cursor + step
        pos[0] = min(max(pos[0], inp.vx), inp.vx + inp.vw - 1)
        pos[1] = min(max(pos[1], inp.vy), inp.vy + inp.vh - 1)
        if np.abs(pos - self.cursor).max() >= 0.5:
            inp.move_to(pos[0], pos[1])
        self.cursor = pos
        self.rel_prev, self.rel_t = smooth.copy(), now

    def _follow_absolute(self, target: np.ndarray, moving: bool) -> None:
        if moving:
            if not self.was_moving and self.cursor is not None:
                self.offset = self.cursor - target  # плавно догоняем вместо скачка
            pos = target + self.offset
            self.offset *= self.cfg["catchup"]
            if np.abs(self.offset).max() < 0.5:
                self.offset[:] = 0
            if self.cursor is None or np.abs(pos - self.cursor).max() >= 1.0:
                self.inp.move_to(pos[0], pos[1])
                self.cursor = pos
        self.was_moving = moving

    def lost(self) -> None:
        self._release()
        self.pinches.reset()
        self.filter.reset()
        self.scroll_last = None
        self.fist_frames = 0
        self.pause_fired = False
        self.was_moving = False
        self.rel_prev = None
        self.engaged = False
        self.mode = "paused" if self.paused else "searching"

    def shutdown(self) -> None:
        self._release()

    # -- main
    def update(self, hand: Hand | None, now: float) -> None:
        if hand is None:
            self.lost()
            return
        cfg = self.cfg
        smooth = self.filter(hand.palm_norm, now)
        is_open = hand.is_open(cfg)
        if is_open:
            self.last_open = now
        # Рука берёт управление только после того, как показала открытую ладонь.
        # Так рука, лежащая на мышке или клавиатуре, ничего не трогает.
        if not self.engaged:
            if not is_open:
                self.mode = "paused" if self.paused else "standby"
                self.was_moving = False
                return
            self.engaged = True
        started, ended = self.pinches.update(hand, cfg, now)
        pinch = self.pinches.active

        # Пауза: щипок большой + мизинец, удерживать pause_hold секунд.
        if pinch == 20 and not self.pause_fired and now - self.pinches.since > cfg["pause_hold"]:
            self.pause_fired = True
            self.toggle_pause()
        if ended == 20:
            self.pause_fired = False

        if self.paused:
            self.mode = "paused"
            self.was_moving = False
            return

        # Щипок большой + указательный. Нажатие откладывается, пока не станет ясно, что это:
        #   отпустили быстро -> левый клик; повели руку -> перетаскивание;
        #   держат неподвижно long_press секунд -> правый клик.
        if started == 8:
            self.pinch_anchor = hand.palm_norm.copy()
            self.drag_unlocked = False
            self.pinch_consumed = False
        elif started == 12:
            self.inp.right_click()
            self.toast("Правый клик")
            self.freeze_until = now + 0.25
        if ended == 8:
            if self.drag_unlocked:
                self._release()
            elif not self.pinch_consumed:
                self.inp.left_down()
                self.inp.left_up()
            self.freeze_until = now + cfg["release_freeze"]

        fist = pinch is None and hand.is_fist(cfg)
        self.fist_frames = self.fist_frames + 1 if fist else 0

        armed = self.scroll_last is not None or now - self.last_open < cfg["fist_arm_time"]
        if self.fist_frames >= cfg["fist_frames"] and armed:
            self.mode = "scroll"
            self._scroll(smooth)
            self._follow(smooth, now, False, hand)
            return
        self.scroll_last = None

        if pinch == 8:
            if not self.drag_unlocked and not self.pinch_consumed:
                if _d(hand.palm_norm, self.pinch_anchor) > cfg["drag_threshold"]:
                    self.drag_unlocked = True
                    self.inp.left_down()  # нажимаем в точке, где начался щипок
                    self.left_down = True
                elif now - self.pinches.since >= cfg["long_press"]:
                    self.pinch_consumed = True
                    self.inp.right_click()
                    self.toast("Правый клик")
            if self.pinch_consumed:
                self.mode = "right"
            else:
                self.mode = "drag" if self.drag_unlocked else "click"
            self._follow(smooth, now, self.drag_unlocked, hand)
            return

        self.mode = "cursor"
        self._follow(smooth, now, now >= self.freeze_until and pinch not in (12, 20), hand)

    def _scroll(self, p: np.ndarray) -> None:
        if self.scroll_last is None:
            self.scroll_last = p.copy()
            self.scroll_acc[:] = 0
            return
        delta = p - self.scroll_last
        self.scroll_last = p.copy()
        sign = 1.0 if self.cfg["natural_scroll"] else -1.0
        speed = self.cfg["scroll_speed"]
        self.scroll_acc += np.array([-delta[0], delta[1]]) * speed * sign
        for axis, horizontal in ((1, False), (0, True)):
            if abs(self.scroll_acc[axis]) >= 40:
                amount = int(self.scroll_acc[axis])
                self.inp.wheel(amount, horizontal=horizontal)
                self.scroll_acc[axis] -= amount


class CommandHand:
    """Вторая рука: громкость, свайпы, медиа, скриншот."""

    def __init__(self, cfg, inp: InputInjector, toast):
        self.cfg, self.inp, self.toast = cfg, inp, toast
        self.pinches = PinchTracker((8, 12))
        self.hist: deque = deque()
        self.cooldown_until = 0.0
        self.vol_anchor = None
        self.fist_frames = 0
        self.fist_fired = False
        self.state = "standby"
        self.engaged = False
        self.last_open = -99.0

    def reset(self) -> None:
        self.pinches.reset()
        self.hist.clear()
        self.vol_anchor = None
        self.fist_frames = 0
        self.fist_fired = False
        self.engaged = False
        self.state = "standby"

    def _action(self, name: str, label: str) -> None:
        combo = self.cfg["actions"].get(name)
        if combo and self.inp.hotkey(combo):
            self.toast(f"{label} ({combo})")

    def update(self, hand: Hand | None, now: float, paused: bool) -> None:
        if hand is None or paused:
            self.reset()
            return
        cfg = self.cfg
        if hand.is_open(cfg):
            self.last_open = now
            self.engaged = True
        if not self.engaged:  # ждём открытую ладонь
            self.state = "standby"
            return
        started, _ = self.pinches.update(hand, cfg, now)
        pinch = self.pinches.active

        # Громкость: щипок указательный + большой, двигать руку вверх/вниз.
        if pinch == 8:
            self.state = "volume"
            y = hand.palm_norm[1]
            if self.vol_anchor is None:
                self.vol_anchor = y
            step = cfg["volume_step"]
            while self.vol_anchor - y >= step:
                self.inp.hotkey("volume_up")
                self.vol_anchor -= step
                self.toast("Громкость +")
            while y - self.vol_anchor >= step:
                self.inp.hotkey("volume_down")
                self.vol_anchor += step
                self.toast("Громкость −")
            self.hist.clear()
            return
        self.vol_anchor = None

        if started == 12:
            self._action("middle_pinch", "Скриншот")

        # Кулак -> одно действие (по умолчанию пауза/воспроизведение).
        if pinch is None and hand.is_fist(cfg):
            self.fist_frames += 1
            self.state = "fist"
            fresh = now - self.last_open < cfg["fist_arm_time"]
            if (self.fist_frames >= cfg["fist_frames"] and not self.fist_fired and fresh
                    and now >= self.cooldown_until):
                self.fist_fired = True
                self.cooldown_until = now + 1.5
                self._action("fist", "Пауза / воспроизведение")
            self.hist.clear()
            return
        self.fist_frames = 0
        self.fist_fired = False

        # Свайпы открытой ладонью.
        if pinch is None and hand.is_open(cfg):
            self.state = "armed"
            self.hist.append((now, hand.palm_norm.copy()))
            while self.hist and now - self.hist[0][0] > cfg["swipe_time"]:
                self.hist.popleft()
            if now >= self.cooldown_until and len(self.hist) >= 3:
                dx, dy = self.hist[-1][1] - self.hist[0][1]
                dist = cfg["swipe_distance"]
                gesture = None
                if abs(dx) > dist and abs(dx) > 1.8 * abs(dy):
                    gesture = "swipe_right" if dx > 0 else "swipe_left"
                elif abs(dy) > dist * 0.8 and abs(dy) > 1.8 * abs(dx):
                    gesture = "swipe_down" if dy > 0 else "swipe_up"
                if gesture:
                    self._action(gesture, SWIPE_NAMES[gesture])
                    self.cooldown_until = now + cfg["swipe_cooldown"]
                    self.hist.clear()
        else:
            self.state = "standby"
            self.hist.clear()


# --------------------------------------------------------------------------- HUD

# Цвета в BGR
WHITE = (255, 255, 255)
MOUSE_COL = (90, 200, 80)      # зелёный — рука-мышь
CMD_COL = (230, 160, 60)       # синий — рука-команды
PAUSE_COL = (150, 150, 150)    # серый — пауза
PINCH_COL = (0, 210, 255)      # жёлтый — щипок
BAR_BG = (38, 38, 38)
TEXT = (235, 235, 235)
MUTED = (160, 160, 160)
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
               (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
               (18, 19), (19, 20), (0, 17)]
SWIPE_NAMES = {"swipe_left": "Свайп влево", "swipe_right": "Свайп вправо",
               "swipe_up": "Свайп вверх", "swipe_down": "Свайп вниз"}
MODE_NAMES = {"cursor": "курсор", "click": "клик (держите — правый клик)", "right": "правый клик", "drag": "перетаскивание", "scroll": "прокрутка",
              "paused": "ПАУЗА", "searching": "руки нет в кадре", "standby": "покажите открытую ладонь"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
BAR_H = 30


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def help_lines(primary: str) -> list[tuple[str, str]]:
    m = "ЛЕВАЯ" if primary == "Left" else "ПРАВАЯ"
    c = "ПРАВАЯ" if primary == "Left" else "ЛЕВАЯ"
    return [
        ("Чтобы начать, покажите открытую ладонь", ""),
        (f"{m} РУКА — МЫШЬ", ""),
        ("  открытая ладонь", "двигать курсор"),
        ("  большой + указательный", "клик (свести и сразу развести)"),
        ("  ... и вести руку", "перетаскивание"),
        ("  ... и держать 0,6 с", "правый клик"),
        ("  большой + средний", "тоже правый клик"),
        ("  кулак + движение", "прокрутка"),
        ("  большой + мизинец, 1 с", "пауза / продолжить"),
        (f"{c} РУКА — КОМАНДЫ", ""),
        ("  большой + указательный", "громкость (вверх / вниз)"),
        ("  большой + средний", "скриншот"),
        ("  кулак", "пауза / воспроизведение"),
        ("  свайп влево / вправо", "сменить рабочий стол"),
        ("  свайп вверх / вниз", "все окна / свернуть всё"),
        ("Чувствительность: F7 / F8 (или − / +)", ""),
        ("F9 пауза, M тачпад/рамка, H помощь, Q выход", ""),
    ]


class HUD:
    def __init__(self, cfg):
        self.cfg = cfg
        self.width = int(cfg["preview_width"])
        self.toasts: deque = deque(maxlen=3)
        self.show_help = False

    def toast(self, msg: str) -> None:
        self.toasts.append((time.monotonic(), msg))
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    @staticmethod
    def _label(img, text, org, col) -> None:
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
        x, y = org
        cv2.rectangle(img, (x - 4, y - th - 5), (x + tw + 4, y + 5), col, -1)
        cv2.putText(img, text, (x, y), FONT, 0.45, WHITE, 1, cv2.LINE_AA)

    def render(self, frame, hands: list[Hand], mouse: MouseController, cmd: CommandHand, fps: float):
        h0, w0 = frame.shape[:2]
        W = self.width
        H = int(round(h0 * W / w0))
        img = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        primary = self.cfg["primary_hand"]

        # Зона управления курсором — тонкая рамка (только в режиме absolute)
        if self.cfg["cursor_mode"] == "absolute":
            x0, y0, x1, y1 = mouse.region()
            cv2.rectangle(img, _pt((x0 * W, y0 * H)), _pt((x1 * W, y1 * H)), (200, 200, 200), 1, cv2.LINE_AA)

        for hand in hands:
            is_primary = hand.label == primary
            col = PAUSE_COL if mouse.paused else (MOUSE_COL if is_primary else CMD_COL)
            pts = hand.norm * np.array([W, H])
            for i, j in CONNECTIONS:
                cv2.line(img, _pt(pts[i]), _pt(pts[j]), WHITE, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, _pt(p), 3, col, -1, cv2.LINE_AA)
            tracker = mouse.pinches if is_primary else cmd.pinches
            if tracker.active is not None:
                cv2.line(img, _pt(pts[4]), _pt(pts[tracker.active]), PINCH_COL, 3, cv2.LINE_AA)
            wrist = _pt(pts[0])
            self._label(img, "Мышь" if is_primary else "Команды", (wrist[0] - 25, min(wrist[1] + 22, H - 6)), col)

        # Строка состояния под картинкой
        bar = np.full((BAR_H, W, 3), BAR_BG, dtype=np.uint8)
        mode = MODE_NAMES.get(mouse.mode, mouse.mode)
        if mouse.mode == "cursor" and mouse.precision:
            mode = "прицеливание"
        kind = "тачпад" if self.cfg["cursor_mode"] == "relative" else "рамка"
        cv2.putText(bar, f"Мышь: {mode}  |  {kind}, чувств. {self.cfg['sensitivity']:g}x", (10, 20), FONT, 0.45,
                    PAUSE_COL if mouse.paused else TEXT, 1, cv2.LINE_AA)
        now = time.monotonic()
        recent = [m for ts, m in self.toasts if now - ts < 2.0]
        right = recent[-1] if recent else f"{fps:.0f} FPS   H — помощь"
        (tw, _), _ = cv2.getTextSize(right, FONT, 0.45, 1)
        cv2.putText(bar, right, (W - tw - 10, 20), FONT, 0.45, TEXT if recent else MUTED, 1, cv2.LINE_AA)
        img = np.vstack([img, bar])

        if self.show_help:
            lines = help_lines(primary)
            box_h = 22 * len(lines) + 16
            overlay = img.copy()
            cv2.rectangle(overlay, (10, 10), (W - 10, 10 + box_h), (30, 30, 30), -1)
            img = cv2.addWeighted(overlay, 0.88, img, 0.12, 0)
            for i, (left, right) in enumerate(lines):
                y = 32 + 22 * i
                header = not left.startswith(" ")
                cv2.putText(img, left.strip(), (22 if header else 30, y), FONT, 0.45,
                            WHITE if header else (200, 200, 200), 1, cv2.LINE_AA)
                if right:
                    cv2.putText(img, right, (int(W * 0.47), y), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return img


# --------------------------------------------------------------------------- main

class KeyWatcher:
    """Отслеживает нажатия клавиш по виртуальным кодам Windows (не зависит от раскладки)."""

    KEYS = {"f7": (0x76,), "f8": (0x77,), "f9": (0x78,), "p": (0x50,), "h": (0x48,), "m": (0x4D,),
            "quit": (0x51, 0x1B), "plus": (0xBB, 0x6B), "minus": (0xBD, 0x6D)}

    def __init__(self):
        self.prev: dict[int, bool] = {}

    def poll(self) -> dict[str, bool]:
        out = {}
        for name, vks in self.KEYS.items():
            hit = False
            for vk in vks:
                down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
                hit |= down and not self.prev.get(vk, False)
                self.prev[vk] = down
            out[name] = hit
        return out


def open_camera(cfg):
    idx = int(cfg["camera_index"])
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera_height"])
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
    raise RuntimeError(f"Не удалось открыть камеру #{idx}. Попробуйте --camera 1 или закройте другие программы с камерой.")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="J.A.R.V.I.S. Hand Control")
    ap.add_argument("--camera", type=int, help="номер камеры (по умолчанию из config.json)")
    ap.add_argument("--dry-run", action="store_true", help="не управлять мышью, только показывать")
    ap.add_argument("--frames", type=int, default=0, help="выйти через N кадров (для теста)")
    ap.add_argument("--right-handed", action="store_true", help="мышь — правой рукой, команды — левой")
    args = ap.parse_args()

    cfg = load_config()
    if args.camera is not None:
        cfg["camera_index"] = args.camera
    if args.right_handed:
        cfg["primary_hand"] = "Right"
    secondary = "Left" if cfg["primary_hand"] == "Right" else "Right"

    set_dpi_aware()
    ensure_model()
    inp = InputInjector(dry_run=args.dry_run)
    hud = HUD(cfg)
    mouse = MouseController(cfg, inp, hud.toast)
    cmd = CommandHand(cfg, inp, hud.toast)

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    cap = open_camera(cfg)

    print("=" * 60)
    print(" J.A.R.V.I.S. Hand Control запущен")
    print(f" Мышь: {'правая' if cfg['primary_hand'] == 'Right' else 'левая'} рука, "
          f"команды: {'левая' if secondary == 'Left' else 'правая'} рука")
    print(" F7/F8 — чувствительность, F9 — пауза (работают в любом окне)")
    print(" В окне программы: H — подсказка, M — тачпад/рамка, Q/Esc — выход")
    if args.dry_run:
        print(" РЕЖИМ DRY-RUN: мышь и клавиатура не управляются")
    print("=" * 60)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    placed = False
    t0 = time.monotonic()
    last_ts = -1
    fps, fps_t, fps_n = 0.0, time.monotonic(), 0
    keys = KeyWatcher()
    frame_count = 0
    fail_count = 0
    hud.toast("Запущено")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fail_count += 1
                if fail_count > 30:
                    raise RuntimeError("Камера перестала отдавать кадры.")
                continue
            fail_count = 0
            if cfg["mirror"]:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.monotonic()
            ts = max(int((now - t0) * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            hands = parse_hands(result, w, h, cfg["mirror"])

            primary_hand = next((hd for hd in hands if hd.label == cfg["primary_hand"]), None)
            secondary_hand = next((hd for hd in hands if hd.label == secondary), None)


            mouse.update(primary_hand, now)
            cmd.update(secondary_hand, now, mouse.paused)

            fps_n += 1
            if now - fps_t >= 0.5:
                fps, fps_t, fps_n = fps_n / (now - fps_t), now, 0

            view = hud.render(frame, hands, mouse, cmd, fps)
            cv2.imshow(WINDOW, view)
            if not placed:
                placed = True
                sw, sh = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
                cv2.moveWindow(WINDOW, sw - view.shape[1] - 20, sh - view.shape[0] - 90)
                if cfg["always_on_top"]:
                    try:
                        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
                    except (cv2.error, AttributeError):
                        pass

            cv2.waitKey(1)
            # Клавиши по физическим кодам: работают при любой раскладке.
            # F7/F8/F9 — в любом окне; остальные — только когда активно окно программы.
            k = keys.poll()
            focused = user32.GetForegroundWindow() == user32.FindWindowW(None, WINDOW)
            if k["f9"] or (focused and k["p"]):
                mouse.toggle_pause()
            step = (k["f8"] or (focused and k["plus"])) - (k["f7"] or (focused and k["minus"]))
            if step:
                cfg["sensitivity"] = round(min(max(cfg["sensitivity"] + 0.25 * step, 0.25), 5.0), 2)
                save_config(cfg)
                hud.toast(f"Чувствительность {cfg['sensitivity']:g}x")
            if focused and k["quit"]:
                break
            if focused and k["h"]:
                hud.show_help = not hud.show_help
            if focused and k["m"]:
                cfg["cursor_mode"] = "absolute" if cfg["cursor_mode"] == "relative" else "relative"
                mouse.was_moving = False
                save_config(cfg)
                hud.toast("Режим: тачпад" if cfg["cursor_mode"] == "relative" else "Режим: рамка")
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            frame_count += 1
            if args.frames and frame_count >= args.frames:
                print(f"Тест завершён: {frame_count} кадров, ~{fps:.0f} FPS")
                break
    finally:
        mouse.shutdown()
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:  # понятное сообщение вместо трейсбэка
        print(f"\n[ОШИБКА] {e}")
        sys.exit(1)
