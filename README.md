# Vision-Arm

A vision-guided robotic arm, built in stages. A camera finds an object, a PyTorch model identifies it, and (later) an arm picks it up. Work in progress.

## Status

Milestone 1 in progress: object classifier with PyTorch on a live webcam feed.

## Roadmap

1. **Vision (PyTorch):** train a small CNN on my own photos of simple objects, run it on a webcam, compare CPU and GPU inference time.
2. **Arm hardware and control:** servo arm driven by my own embedded C code (PWM).
3. **Calibration:** map camera pixels to arm positions.
4. **CUDA:** write GPU kernels for the vision step and measure the speedup.
5. **Verilog (optional):** servo controller with PID in fixed-point hardware.

## Results

| Milestone | Result |
|---|---|
| 1. Classifier | 10/10 on a 10-photo held-out test set (54 photos, 2 objects, one desk, so a small and easy test). CPU vs GPU timing: not yet measured |

## Layout

- `data/raw/`: my object photos (not committed)
- `notebooks/`: Colab notebooks
- `src/`: scripts
- `docs/`: notes and diagrams

## Running

Setup instructions will be added once milestone 1 works.
