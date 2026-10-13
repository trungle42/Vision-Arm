# Vision-Arm

A vision-guided robotic arm, built in stages. A camera finds an object, a PyTorch model identifies it, and (later) an arm picks it up. Work in progress.

## Status

Milestone 1 works: a PyTorch classifier runs live on a webcam feed, labels what it sees as cube, pen, or nothing, and marks where the object is in the frame. A CUDA preprocessing kernel is written and measured (milestone 4). The arm itself is not built yet: the servo code, inverse kinematics, and calibration exist only as tested code and simulation, with no hardware attached.

## Roadmap

1. **Vision (PyTorch):** train a small CNN on my own photos of simple objects, run it on a webcam, compare CPU and GPU inference time. *Done.*
2. **Arm hardware and control:** servo arm driven by my own embedded C code (PWM). *In progress: servo module in C is tested on a PC, 2-joint inverse kinematics is simulated, no servos yet.*
3. **Calibration:** map camera pixels to arm positions. *In progress: homography code is tested on simulated points only. It needs real measurements.*
4. **CUDA:** write GPU kernels for the vision step and measure the speedup. *Done for the preprocessing step, see results.*
5. **Verilog (optional):** servo controller with PID in fixed-point hardware. *Not started.*

## Results

| Milestone | Result |
|---|---|
| 1. Classifier | 10/10 on a 10-photo held-out test set (54 photos, 2 objects, one desk, so a small and easy test). Inference time per photo (batch size 1, model only, 3 runs): GPU (Tesla T4) 4.4 to 4.6 ms, CPU (2-core Xeon on Colab) 57 to 77 ms, about 13x to 17x faster |
| 1b. 3-class retrain (cube, pen, nothing) | 12/12 on a 12-photo held-out set (63 photos), still a small and easy test. The live webcam demo was only checked by eye, not measured |
| 1c. Inference on my laptop | Apple M5 Pro, batch size 1, model only, 3 runs: CPU 8.4 to 12.3 ms, Apple GPU (MPS) 3.0 to 4.3 ms per photo |
| 4. CUDA preprocessing | One kernel does BGR to RGB, bilinear resize (640x480 to 224x224), scaling, and HWC to CHW. Matches a CPU reference (max abs difference 1.2e-7). Tesla T4, 3 runs: kernel alone 0.009 to 0.012 ms per frame vs 1.56 to 1.70 ms for my single-threaded CPU loop (about 130x to 190x). Including the host-to-device copy of the frame: 0.265 to 0.281 ms (about 6x). The baseline is my own naive loop, not OpenCV, and the test frame is random data |

## Layout

- `data/raw/`: my object photos (not committed)
- `notebooks/`: Colab notebooks
- `src/`: `capture.py` (save webcam photos), `train.py` (train the classifier), `classify.py` (check one photo), `webcam.py` (live demo)
- `src/cuda/`: CUDA preprocessing kernel with a CPU check and timing
- `src/servo/`: servo pulse-width and speed-limit code in C, with tests
- `src/arm/`: inverse kinematics, pixel-to-table calibration, and the pipeline joining them (simulation, placeholder values)
- `models/`: trained weights (not committed) and class names
- `docs/`: notes and diagrams

## Running

Requires Python 3 with `torch`, `torchvision`, and `opencv-python`. Run everything from the repo root.

```bash
python src/capture.py pen     # press space to save a photo, q to quit; repeat for cube and nothing
python src/train.py           # trains on data/raw and saves models/model.pth
python src/webcam.py          # live demo: b saves an empty-scene reference for object position, q quits
```

Servo tests (C, no hardware needed): `cd src/servo && make test`.
CUDA kernel (needs an NVIDIA GPU and nvcc): `nvcc -O2 src/cuda/preprocess.cu -o preprocess && ./preprocess`.

On macOS the terminal app needs camera permission.
