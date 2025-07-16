# Model Deploy Execution Attack Lab

This project is an automated adversarial attack experimentation platform targeting the security of **binary-level AI model deployments**. It supports multiple mainstream inference engines (e.g., MNN, NCNN, ONNXRuntime) and various attack algorithms.

## Core Features

- **Gray-box (State-matching) & Black-box (Decision-based) Attacks**: Supports both gray-box attacks, which leverage internal model states, and black-box attacks that only rely on final model outputs.
- **Source-Free State Extraction**: Automatically extracts internal model states via GDB hooks, eliminating the need for source code.
- **Automated Input Adaptation**: Automatically adapts image channel counts and dimensions to match model requirements, compatible with both color and grayscale inputs.
- **Optimized for Performance**: Features multi-process parallelization, memory optimization, and detailed logging for efficient and transparent execution.

## Supported Attack Algorithms

- **CMA-ES**: A gray-box (state-matching) algorithm ideal for low- to medium-dimensional inputs- It uses internal model states obtained via GDB hooks to guide its optimization process.
- **NES (Natural Evolution Strategies)**: A gray-box (state-matching) algorithm suitable for high-dimensional inputs- It has low memory consumption and estimates gradients using internal states from GDB hooks.
- **Boundary Attack, HopSkipJump, Sign-OPT**: Black-box (decision-based) algorithms that are effective when only the final decision (true/false) of the model is of interest.

## Setup Guide

### 1. Prerequisites

- **Compile C++ Code**: Compile the C++ source code using the provided `CMakeLists-txt` file.
  ```bash
  mkdir build
  cd build
  cmake ..
  make
  ```
- **Organize Assets**:
  - Move compiled executables (e.g., `emotion_ferplus_mnn`) to `resources/execution_files/`.
  - Place model files (`-onnx`, `-mnn`, etc.) in `resources/models/`.
  - Store images for analysis in `resources/images/`.

### 2. Install Dependencies

Install system libraries and Python packages using the provided script (on Ubuntu 24.04).
```bash
bash scripts/install_dependencies.sh
```
Activate the Python virtual environment:
```bash
source scripts/.venv/bin/activate
```

## Usage Examples

Here are typical commandline examples for running attacks on the `ssrnet_age_mnn` model.

### NES Attack (Gray-box, State-matching)
```bash
python3 src/attackers/nes_attack.py \
    --executable assets/bin/ssrnet_age_mnn \
    --image assets/test_lite_age_googlenet_old2.jpg \
    --hooks configs/hook_ssrnet.json \
    --model assets/ssrnet.mnn \
    --golden-image assets/test_lite_age_googlenet.jpg \
    --output-dir outputs/nes_attack_1 \
    --iterations 10000 \
    --learning-rate 10.0 \
    --lr-decay-rate 0.95 \
    --lr-decay-steps 20 \
    --population-size 100 \
    --sigma 0.2 \
    --workers 14 \
    --enable-stagnation-decay \
    --stagnation-patience 10
```

### CMA-ES Attack (Gray-box, State-matching)
```bash
python3 src/attackers/cmaes_attack.py \
    --executable resources/execution_files/mnist_mnn \
    --model resources/models/mnist.mnn \
    --hooks hook_config/mnist_mnn_hook_config.json \
    --golden-image resources/images/mnist_sample/7/7_0.png \
    --image resources/images/mnist_sample/0/0_0.png \
    --output-dir outputs/cmaes_attack_1 \
    --iterations 100 \
    --population-size 100 \
    --sigma 5.0 \
    --l-inf-norm 16.0
```

> For other attack algorithms, please refer to the help message of each script via the `-h` flag.

## Model & Input Requirements

| Model                  | Input Dimensions | Channels | Data Type |
| :--------------------- | :--------------- | :------- | :-------- |
| `ssrnet_age_mnn`       | 64x64            | 3        | float32   |
| `emotion_ferplus_mnn`  | 64x64            | 1        | float32   |
| `gender_googlenet_mnn` | 224x224          | 3        | float32   |
| `example_mnn`          | 224x224          | 3        | float32   |
| `ultraface_detector`   | 320x240          | 3        | float32   |
| `yolov5_detector`      | 640x640          | 3        | float32   |

- The scripts will automatically detect and adapt the input image's dimensions and channel count to match the model's requirements.
- It is recommended that the input image and the golden image have the same dimensions and channel count.

## Frequently Asked Questions (FAQ)

- **Memory Overflow**: CMA-ES can be memory-intensive with high-resolution images. It is advisable to resize images to smaller dimensions (e.g., 64x64, 128x128) first.
- **Image Format**: Ensure images are in a common format (JPEG/PNG) and can be read by OpenCV.
- **Hook Configuration**: The `hook_xx.json` file must match the version of the executable to ensure GDB can hit breakpoints correctly.
- **GDB Permissions**: If GDB fails to attach, check the `/proc/sys/kernel/yama/ptrace_scope` setting.
- **Decision-based Attacks**: These attacks require an original image (classified as `false`) and a starting adversarial image (classified as `true`), both of which must have identical dimensions.

## Contributing

We welcome contributions! Please submit an issue or pull request, or contact us via email.

## Original Image Classification Script

The project also includes a utility to classify images using the deployed models. After setup, you can use `scripts/process_images.sh` to sort images based on model predictions.

### Running the Classification Script

First, grant execute permissions:
```bash
chmod +x scripts/process_images.sh
```
Then, run the script:
```bash
./scripts/process_images.sh
```

The script will:
1. Iterate through all executables in `resources/execution_files/`.
2. Create a corresponding subdirectory in the `results/` directory for each executable.
3. Create `true` and `false` subfolders within each subdirectory.
4. Process each image in `resources/images/` and copy it to the `true` or `false` folder based on the executable's output.

### Viewing Classification Results

The classified images will be organized in the `results/` directory as follows:
```
results/
├── emotion_ferplus_mnn/
│   ├── true/
│   │   ├── image1.jpg
│   │   └── --
│   └── false/
│       ├── image2.jpg
│       └── --
└── fsanet_headpose_onnxruntime/
    ├── true/
    │   └── --
    └── false/
        └── --
```





