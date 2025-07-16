# Model Deploy Execution Attack Lab

本项目是一个面向**AI模型二进制部署安全**的自动化对抗攻击实验平台，支持多种主流推理引擎（MNN、NCNN、ONNXRuntime等）和多种攻击算法。核心特性包括：
- 支持**灰盒（状态匹配）**与**黑盒（决策型）**攻击
- 通过GDB hook自动提取模型内部状态，无需源码
- 自动适配图片通道数/尺寸，兼容彩色与灰度模型
- 支持多进程并行、内存优化、详细日志输出

## 支持的攻击算法
- **CMA-ES**：灰盒（状态匹配），适合低/中维度输入，通过GDB hook获取模型内部状态进行优化
- **NES**：灰盒（状态匹配），适合高维输入，内存消耗低，通过GDB hook获取模型内部状态进行梯度估计
- **Boundary Attack**、**HopSkipJump**、**Sign-OPT**：黑盒（决策型），适合只关心最终决策的场景，仅使用模型的输出结果（true/false）

## 典型用法（以ssrnet_age_mnn为例）

### NES攻击（灰盒，状态匹配）
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

### CMA-ES攻击（灰盒，状态匹配）
```bash
python3 src/attackers/cmaes_attack.py \
    --executable assets/bin/ssrnet_age_mnn \
    --model assets/ssrnet.mnn \
    --hooks configs/hook_ssrnet.json \
    --golden-image assets/test_lite_age_googlenet.jpg \
    --image assets/test_lite_age_googlenet_old2.jpg \
    --output-dir outputs/cmaes_attack_1 \
    --iterations 100 \
    --population-size 100 \
    --sigma 5.0 \
    --l-inf-norm 16.0
```

> 其他攻击算法用法类似，详见各脚本`-h`帮助。

## 输入图片和模型要求

| 模型                 | 输入尺寸  | 通道数 | 数据类型 |
| :------------------- | :-------- | :----- | :------- |
| ssrnet_age_mnn       | 64 x 64   | 3      | float32  |
| emotion_ferplus_mnn  | 64 x 64   | 1      | float32  |
| gender_googlenet_mnn | 224 x 224 | 3      | float32  |
| example_mnn          | 224 x 224 | 3      | float32  |
| ultraface_detector   | 320 x 240 | 3      | float32  |
| yolov5_detector      | 640 x 640 | 3      | float32  |

- 脚本会自动检测图片通道数/尺寸并适配到模型要求。
- 建议输入图片与golden-image尺寸、通道一致。

## 常见问题与建议
- **内存溢出**：CMA-ES在高分辨率下会极度吃内存，建议先将图片resize到较小尺寸（如64x64、128x128）。
- **图片格式**：务必保证图片为常见格式（jpg/png），且能被OpenCV正常读取。
- **hook配置**：`hook_xxx.json`必须与当前可执行文件版本匹配，否则GDB无法命中断点。
- **GDB权限**：如遇GDB无法attach，需检查`/proc/sys/kernel/yama/ptrace_scope`设置。
- **决策型攻击**：需提供一张原始图片（判定为false）和一张对抗起点图片（判定为true），尺寸必须一致。

## 贡献与联系方式
- 欢迎提交issue、PR或邮件交流。
- 联系方式：请见项目主页或相关文档。

This is the repository for our paper "Evading Deep-Learning-based anit-virus Scanners".

## Usage Guide

This project is designed to run models through different deep learning inference engines (such as MNN, NCNN, ONNX Runtime, TNN) and classify images based on the model's output.

### 1. Prerequisites

Before running the project, please ensure you have met the following conditions:

*   **Compile C++ Code**: The C++ source code in the project needs to be compiled. Please use the provided `CMakeLists.txt` file for compilation.
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```
*   **Prepare Executables**: Move all compiled executables (e.g., `emotion_ferplus_mnn`, `fsanet_headpose_onnxruntime`, etc.) to the `resources/execution_files/` directory.
*   **Prepare Model Files**: Ensure all model files (`.onnx`, `.mnn`, `.param`, `.bin`, `.tnnproto`, `.tnnmodel`, etc.) are located in the `resources/models/` directory.
*   **Prepare Images**: Place the image files to be analyzed and classified in the `resources/images/` directory.

### 2. Install Dependencies

The project depends on certain system libraries and Python packages. You can run the `install_dependencies.sh` script to install them automatically (currently supports Ubuntu 24.04).

```bash
bash scripts/install_dependencies.sh
```
After installation, remember to activate the Python virtual environment:
```bash
source scripts/.venv/bin/activate
```

### 3. Run the Image Processing Script

Once all files and dependencies are ready, you can run the `process_images.sh` script to start processing.

First, grant execute permissions to the script:
```bash
chmod +x scripts/process_images.sh
```

Then, run the script directly:
```bash
./scripts/process_images.sh
```

The script will automatically perform the following actions:
1.  Iterate through all executables in the `resources/execution_files/` directory.
2.  For each executable, create a corresponding subdirectory in the `results/` directory.
3.  Within each subdirectory, create two folders: `true` and `false`.
4.  Process each image in the `resources/images/` directory using the executable.
5.  Based on the program's output (`true` or `false`), copy the original image into the corresponding `true` or `false` folder.

### 4. View Results

After processing is complete, you can find the classified images in the `results/` directory. The directory structure will look like this:

```
results/
├── emotion_ferplus_mnn/
│   ├── true/
│   │   ├── image1.jpg
│   │   └── ...
│   └── false/
│       ├── image2.jpg
│       └── ...
└── fsanet_headpose_onnxruntime/
    ├── true/
    │   └── ...
    └── false/
        └── ...
```





