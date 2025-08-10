# Model Deploy Execution Attack Lab

This project is an automated adversarial attack experimentation platform targeting the security of **binary-level AI model deployments**. It supports multiple mainstream inference engines (e.g., MNN, NCNN, ONNXRuntime) and various attack algorithms.

## 🚀 项目架构 (Project Architecture)

```
.
├── hook_config/         # GDB 钩子的 JSON 配置文件
├── outputs/             # 攻击过程中生成的对抗样本和日志
├── pre_attack_scripts/  # 攻击前的准备脚本 (如筛选 "false" 图片)
├── resources/           # 存放所有攻击所需的资源
│   ├── execution_files/ # 编译好的模型可执行程序
│   ├── false_image_list/# "false" 图片的列表文件
│   ├── images/          # 用于测试和攻击的原始图片
│   └── models/          # AI 模型文件 (.onnx, .mnn, etc.)
├── results/             # 批量图像分类的结果
├── scripts/             # 辅助脚本 (如环境安装)
└── src/                 # 攻击算法的核心 Python 源码
```

## 📊 完整工作流程 (End-to-End Workflow)

Follow these steps to set up the environment and run an attack.

### 步骤 1：环境设置 (Setup Environment)

1.  **编译 C++ 代码**:
    Compile the C++ source code using the provided `CMakeLists.txt` file.
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

2.  **安装依赖**:
    Install system libraries and Python packages using the provided script (tested on Ubuntu 24.04).
    ```bash
    bash scripts/install_dependencies.sh
    ```
    Activate the Python virtual environment:
    ```bash
    source scripts/.venv/bin/activate
    ```

### 步骤 2：准备资源文件 (Prepare Assets)

- Move compiled executables (e.g., `emotion_ferplus_mnn`) to `resources/execution_files/`.
- Place model files (`.onnx`, `.mnn`, etc.) in `resources/models/`.
- Store images for analysis in `resources/images/`.

### 步骤 3：(可选但推荐) 筛选攻击候选图片 (Filter Attack Candidates)

To improve attack efficiency, you can first find images that the model already classifies as "false". These images are often better starting points for an attack.

The `generate_false_image_list.sh` script automates this process.

1.  **授权**:
    ```bash
    chmod +x pre_attack_scripts/generate_false_image_list.sh
    ```

2.  **运行脚本**:
    Provide the path to the executable you want to test.
    ```bash
    # Example for the emotion_ferplus_mnn model
    ./pre_attack_scripts/generate_false_image_list.sh resources/execution_files/emotion_ferplus_mnn
    ```
    The script will generate a file like `resources/false_image_list/emotion_ferplus_mnn_false_list.txt`, containing paths to all images classified as "false".

### 步骤 4：执行攻击 (Run an Attack)

Here are typical command-line examples for running attacks.

#### NES Attack (Gray-box, State-matching)
```bash
python3 src/attackers/nes_attack.py \
    --executable resources/execution_files/mnist_mnn \
    --model resources/models/mnist.mnn \
    --hooks hook_config/mnist_mnn_hook_config.json \
    --golden-image resources/images/mnist_sample/7/7_0.png \
    --image outputs/nes_attack_1/best_attack_image_nes_host.png \
    --output-dir outputs/nes_attack_1 \
    --iterations 200 \
    --learning-rate 20.0 \
    --population-size 200
```

#### CMA-ES Attack (Gray-box, State-matching)
```bash
python3 src/attackers/cmaes_attack.py \
    --executable resources/execution_files/mnist_mnn \
    --model resources/models/mnist.mnn \
    --hooks hook_config/mnist_mnn_hook_config.json \
    --golden-image resources/images/mnist_sample/7/7_0.png \
    --image resources/images/mnist_sample/0/0_0.png \
    --output-dir outputs/cmaes_attack_1 \
    --iterations 100 \
    --population-size 100
```

> For other attack algorithms, please refer to the help message of each script via the `-h` flag.

## 🔬 支持的攻击算法 (Supported Attack Algorithms)

- **CMA-ES**: A gray-box (state-matching) algorithm ideal for low- to medium-dimensional inputs. It uses internal model states obtained via GDB hooks to guide its optimization process.
- **NES (Natural Evolution Strategies)**: A gray-box (state-matching) algorithm suitable for high-dimensional inputs. It has low memory consumption and estimates gradients using internal states from GDB hooks.
- **Boundary Attack, HopSkipJump, Sign-OPT**: Black-box (decision-based) algorithms that are effective when only the final decision (true/false) of the model is of interest.

## 🪝 GDB 钩子配置 (GDB Hook Configuration)

The `hook_config/` directory contains JSON files that tell the attack scripts where to set GDB breakpoints to extract intermediate features from the model executable. You must ensure the `hook_xx.json` file matches the version of the executable to ensure GDB can hit breakpoints correctly.

## ❓ 常见问题 (FAQ)

- **Memory Overflow**: CMA-ES can be memory-intensive with high-resolution images. It is advisable to resize images to smaller dimensions (e.g., 64x64, 128x128) first.
- **Image Format**: Ensure images are in a common format (JPEG/PNG) and can be read by OpenCV.
- **GDB Permissions**: If GDB fails to attach, check the `/proc/sys/kernel/yama/ptrace_scope` setting.
- **Decision-based Attacks**: These attacks require an original image (classified as `false`) and a starting adversarial image (classified as `true`), both of which must have identical dimensions.





