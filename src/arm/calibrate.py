import json

import cv2
import numpy as np


def fit(pixel_pts, table_pts):
    """Fit the pixel -> table (cm) mapping from matching points. Needs at least 4."""
    H, _ = cv2.findHomography(np.array(pixel_pts, np.float32), np.array(table_pts, np.float32))
    return H


def pixel_to_table(H, px, py):
    p = H @ np.array([px, py, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def save(H, path="models/calibration.json"):
    json.dump(H.tolist(), open(path, "w"))


def load(path="models/calibration.json"):
    return np.array(json.load(open(path)))


if __name__ == "__main__":
    # Simulated camera: a made-up tilted view of a 30 x 20 cm table.
    table_corners = np.float32([[0, 0], [30, 0], [30, 20], [0, 20]])
    pixel_corners = np.float32([[120, 90], [540, 110], [580, 400], [80, 380]])
    table_to_pixel = cv2.getPerspectiveTransform(table_corners, pixel_corners)

    def project(x, y):
        p = table_to_pixel @ np.array([x, y, 1.0])
        return p[0] / p[2], p[1] / p[2]

    rng = np.random.default_rng(0)
    # calibration points: 6 spots, each pixel reading is off by about 1 pixel
    cal_table = [(2, 2), (28, 3), (27, 18), (3, 17), (15, 10), (15, 3)]
    cal_pixel = [np.array(project(*t)) + rng.normal(0, 1.0, 2) for t in cal_table]
    H = fit(cal_pixel, cal_table)

    # test on points that were not used for calibration
    errs = []
    for _ in range(200):
        tx, ty = rng.uniform(1, 29), rng.uniform(1, 19)
        mx, my = pixel_to_table(H, *project(tx, ty))
        errs.append(np.hypot(mx - tx, my - ty))
    print(f"mean error {np.mean(errs):.3f} cm, worst {np.max(errs):.3f} cm")
    assert np.max(errs) < 0.5
    print("calibration test passed")
