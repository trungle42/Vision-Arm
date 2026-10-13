"""Camera pixel -> table cm -> joint angles -> servo pulse widths."""
import ctypes
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from arm import calibrate, ik

SERVO_DIR = Path(__file__).resolve().parent.parent / "servo"
LIB_NAME = "libservo.dylib" if sys.platform == "darwin" else "libservo.so"

# Where the arm base sits in table coordinates (cm). Placeholder until there is a real arm.
BASE_X, BASE_Y = 15.0, -3.0


class Servo(ctypes.Structure):  # must match servo_t in servo.h
    _fields_ = [("min_deg", ctypes.c_float), ("max_deg", ctypes.c_float),
                ("min_us", ctypes.c_uint16), ("max_us", ctypes.c_uint16),
                ("max_speed_dps", ctypes.c_float), ("current_deg", ctypes.c_float)]


def _load_servo_lib():
    lib_path = SERVO_DIR / LIB_NAME
    if not lib_path.exists():
        subprocess.run(["make", "lib"], cwd=SERVO_DIR, check=True, capture_output=True)
    lib = ctypes.CDLL(str(lib_path))
    lib.servo_init.argtypes = [ctypes.POINTER(Servo), ctypes.c_float, ctypes.c_float,
                               ctypes.c_uint16, ctypes.c_uint16, ctypes.c_float]
    lib.servo_angle_to_us.argtypes = [ctypes.POINTER(Servo), ctypes.c_float]
    lib.servo_angle_to_us.restype = ctypes.c_uint16
    return lib


_lib = _load_servo_lib()
_servos = [Servo(), Servo()]
for _s in _servos:
    _lib.servo_init(ctypes.byref(_s), 0.0, 180.0, 500, 2500, 90.0)


def placeholder_calibration():
    """Made-up tilted view of a 30 x 20 cm table, used until a real calibration is saved."""
    pixel_corners = np.float32([[120, 90], [540, 110], [580, 400], [80, 380]])
    table_corners = np.float32([[0, 0], [30, 0], [30, 20], [0, 20]])
    return cv2.getPerspectiveTransform(pixel_corners, table_corners)


def load_calibration():
    path = Path("models/calibration.json")
    return calibrate.load(str(path)) if path.exists() else placeholder_calibration()


def pixel_to_pulses(H, px, py):
    """Returns (table_xy, joint_degrees, pulse_us), or None if the arm cannot reach it."""
    tx, ty = calibrate.pixel_to_table(H, px, py)
    sol = ik.inverse(tx - BASE_X, ty - BASE_Y)
    if sol is None:
        return None
    degs = [math.degrees(q) for q in sol]
    pulses = [int(_lib.servo_angle_to_us(ctypes.byref(s), d)) for s, d in zip(_servos, degs)]
    return (tx, ty), degs, pulses


if __name__ == "__main__":
    H = placeholder_calibration()
    # the pixel at the middle of the table should land near the middle (15, 10) cm
    tx, ty = calibrate.pixel_to_table(H, *cv2.perspectiveTransform(
        np.float32([[[15, 10]]]), np.linalg.inv(H))[0][0])
    assert abs(tx - 15) < 1e-3 and abs(ty - 10) < 1e-3

    out = pixel_to_pulses(H, 330, 245)
    print("middle of frame:", out)
    assert out is not None and all(500 <= p <= 2500 for p in out[2])

    far = pixel_to_pulses(H, 80, 380)  # table (0, 20): about 27 cm from the base, beyond the 20 cm reach
    print("far corner:", far)
    assert far is None
    print("pipeline test passed")
