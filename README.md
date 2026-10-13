# Vision-Arm

A vision-guided robotic arm, built in stages. A camera finds an object, a PyTorch model identifies it, and (later) an arm picks it up. Work in progress.

## Status

Milestone 1 works: a PyTorch classifier runs live on a webcam feed and labels what it sees as cube, pen, or nothing. Still to do: more varied photos, then the arm (milestone 2).

## Roadmap

1. **Vision (PyTorch):** train a small CNN on my own photos of simple objects, run it on a webcam, compare CPU and GPU inference time.
2. **Arm hardware and control:** servo arm driven by my own embedded C code (PWM).
3. **Calibration:** map camera pixels to arm positions.
4. **CUDA:** write GPU kernels for the vision step and measure the speedup.
5. **Verilog (optional):** servo controller with PID in fixed-point hardware.

## Results

| Milestone | Result |
|---|---|
| 1. Classifier | 10/10 on a 10-photo held-out test set (54 photos, 2 objects, one desk, so a small and easy test). Inference time per photo (batch size 1, model only, 3 runs): GPU (Tesla T4) 4.4 to 4.6 ms, CPU (2-core Xeon on Colab) 57 to 77 ms, about 13x to 17x faster |
| 1b. 3-class retrain (cube, pen, nothing) | 12/12 on a 12-photo held-out set (63 photos), still a small and easy test. The live webcam demo was only checked by eye, not measured |

## Layout

- `data/raw/`: my object photos (not committed)
- `notebooks/`: Colab notebooks
- `src/`: `capture.py` (save webcam photos), `train.py` (train the classifier), `classify.py` (check one photo), `webcam.py` (live demo)
- `models/`: trained weights (not committed) and class names
- `docs/`: notes and diagrams

## Running

Requires Python 3 with `torch`, `torchvision`, and `opencv-python`. Run everything from the repo root.

```bash
python src/capture.py pen     # press space to save a photo, q to quit; repeat for cube and nothing
python src/train.py           # trains on data/raw and saves models/model.pth
python src/webcam.py          # live demo, press q to quit
```

On macOS the terminal app needs camera permission.
