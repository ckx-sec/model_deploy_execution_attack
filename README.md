# 情绪识别模型攻击实验室 (重构版)

本项目是一个功能强大的框架，用于对作为原生二进制文件运行的机器学习模型进行高级对抗性攻击。它专注于灰盒和白盒攻击，利用 GDB 等调试工具来监测目标应用程序并提取内部状态信息，而无需修改原始可执行文件。

这种方法可以对只有编译后二进制文件的应用程序（例如移动应用或专有软件中的程序）进行高度精确和有效的攻击。

## 核心概念

本实验室围绕“神谕”（Oracle）这一概念构建。神谕即是我们的目标应用程序，它在 GDB 的监控下运行。攻击脚本使用生成的输入（图像）查询神谕，并根据神谕的反馈来指导攻击方向。反馈机制的类型决定了攻击的类型：

1.  **基于决策的攻击 (灰盒)**: 神谕只返回一个 `true`/`false` 状态，表示输入图像是否触发了特定的、期望的行为（例如，错误的分类）。这类攻击的目标是找到一个与原始图像差异最小的对抗性样本。
    *   已实现算法: `boundary_attack.py`, `hopskipjump_attack.py`, `sign_opt_attack.py`

2.  **基于状态匹配的攻击 (白盒)**: 神谕返回一个详细的内部状态向量（例如，某个神经网络层的输出）。这类攻击的目标是扰动一张输入图像，使其能够让应用程序复现一个由“黄金”目标图像所产生的**确切**内部状态。
    *   已实现算法: `cmaes_attack.py`, `nes_attack.py`, `spsa_attack.py`

## 目录结构

项目已被重构为以下结构：

```
emotion_attack_lab_refactored/
├── README.md                # 本文档
├── CMakeLists.txt           # 主项目 CMake 文件
├── cmake/                   # 其他 CMake 模块
├── third_party/             # 预编译的第三方推理引擎 (MNN, NCNN, ONNXRuntime 等)
│
├── bin/                     # 存放编译好的目标可执行文件
│   └── mnist_mnn_console
│
├── assets/                  # 模型、测试图像和目标图像
├── configs/                 # Hook 配置文件 (*.hooks.json)
├── outputs/                 # 保存攻击结果 (图像、日志) 的目录
├── scripts/                 # 辅助脚本，主要是 GDB 运行器
│   └── run_gdb_host.sh
│
└── src/
    └── attackers/           # 存放不同攻击算法的核心 Python 脚本
        ├── boundary_attack.py
        ├── cmaes_attack.py
        └── ... and others
```

## 如何运行攻击

所有攻击都通过 `src/attackers/` 中的相应 Python 脚本启动。它们共享相似的命令行接口。

### 白盒攻击 (状态匹配)

#### NES (Natural Evolution Strategies)
一种使用自然演化策略的黑盒优化算法，通过评估一群扰动的适应度来估计梯度。
```bash
python3.10 src/attackers/nes_attack.py \
    --executable bin/mnist_mnn_console \
    --image assets/test_digit_2.png \
    --hooks configs/hooks.json \
    --model assets/mnist.mnn \
    --golden-image assets/test_digit_7.png \
    --output-dir outputs/nes_attack_1 \
    --iterations 10000 \
    --learning-rate 10.0 \
    --lr-decay-rate 0.95 \
    --lr-decay-steps 20 \
    --population-size 500 \
    --sigma 0.2 \
    --workers 14 \
    --enable-stagnation-decay \
    --stagnation-patience 10
```

#### CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
一种强大的演化算法，它通过学习扰动之间的协方差矩阵来高效地探索搜索空间。
```bash
python3.10 src/attackers/cmaes_attack.py \
    --executable bin/mnist_mnn_console \
    --model assets/mnist.mnn \
    --hooks configs/hooks.json \
    --golden-image assets/test_digit_7.png \
    --image assets/test_digit_2.png \
    --output-dir outputs/cmaes_attack_1 \
    --iterations 500 \
    --population-size 100 \
    --sigma 5.0 \
    --l-inf-norm 16.0
```

#### SPSA (Simultaneous Perturbation Stochastic Approximation)
一种随机梯度近似算法，每次迭代仅需两次评估即可估计梯度，查询效率非常高。
```bash
python3.10 src/attackers/spsa_attack.py \
    --executable bin/mnist_mnn_console \
    --image assets/test_digit_2.png \
    --hooks configs/hooks.json \
    --model assets/mnist.mnn \
    --golden-image assets/test_digit_7.png \
    --output-dir outputs/spsa_attack_1 \
    --iterations 10000 \
    --learning-rate 2.0 \
    --spsa-c 0.1 \
    --workers 14 \
    --enable-lr-decay \
    --stagnation-patience 25 \
    --lr-decay-rate 0.9
```

---

### 灰盒攻击 (决策型)

#### Boundary Attack
一个非常经典的方法。它从一个已经判定为 true 的大型扰动样本开始，然后沿着决策边界（decision boundary）逐步向原始样本移动，以在保持 true 结果的同时最小化扰动。
```bash
python3.10 src/attackers/boundary_attack.py \
    --executable bin/mnist_mnn_console \
    --model assets/mnist.mnn \
    --hooks configs/hooks.json \
    --image assets/test_digit_2.png \
    --start-adversarial assets/test_digit_7.png \
    --iterations 500 \
    --source-step 0.01 \
    --spherical-step 0.01 \
    --output-dir outputs/boundary_attack_1
```

#### HopSkipJumpAttack
目前最先进（State-of-the-art）的决策型攻击之一，它在查询效率和扰动大小上通常表现优于 Boundary Attack。
```bash
python3.10 src/attackers/hopskipjump_attack.py \
    --executable bin/mnist_mnn_console \
    --model assets/mnist.mnn \
    --hooks configs/hooks.json \
    --image assets/test_digit_2.png \
    --start-adversarial assets/test_digit_7.png \
    --iterations 50 \
    --num-grad-queries 100 \
    --binary-search-steps 10 \
    --output-dir outputs/hopskip_attack_1
```

#### Sign-OPT
也是一种高效的决策型攻击算法，通过二分搜索和符号估计来优化。
```bash
python3.10 src/attackers/sign_opt_attack.py \
    --executable bin/mnist_mnn_console \
    --model assets/mnist.mnn \
    --hooks configs/hooks.json \
    --image assets/test_digit_2.png \
    --start-adversarial assets/test_digit_7.png \
    --iterations 500 \
    --alpha 0.2 \
    --step-size 1.0 \
    --binary-search-steps 10 \
    --output-dir outputs/signopt_attack_1
```

### 关键参数说明:

*   `--executable`: 编译好的、运行模型推理的 C++ 应用程序的路径。
*   `--model`: ML 模型文件的路径 (例如 `.onnx`, `.mnn`)。
*   `--hooks`: 定义 GDB 应挂钩以提取内部状态的内存地址的 JSON 文件的路径。
*   `--image`: 原始的、非对抗性的输入图像。
*   `--golden-image`: (用于状态匹配攻击) 我们希望复现其内部状态的目标图像。
*   `--start-adversarial`: (用于决策型攻击) 已知具有对抗性效果的初始图像。
*   `--output-dir`: 保存生成的对抗性图像和日志的目录。
*   `--iterations`: 运行的攻击迭代次数。 





```
python3 src/attackers/nes_attack.py     --executable assets/bin/ssrnet_age_mnn     --image assets/test_lite_age_googlenet_old2.jpg     --hooks configs/hook_ssrnet.json     --model assets/ssrnet.mnn     --golden-image assets/test_lite_age_googlenet.jpg     --output-dir outputs/nes_attack_1     --iterations 10000     --learning-rate 10.0     --lr-decay-rate 0.95     --lr-decay-steps 20     --population-size 100     --sigma 0.2     --workers 14     --enable-stagnation-decay     --stagnation-patience 10

```

```
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

| 模型                 | 输入尺寸  | 通道数 | 数据类型 |
| :------------------- | :-------- | :----- | :------- |
| ssrnet_age_mnn       | 64 x 64   | 3      | float32  |
| emotion_ferplus_mnn  | 64 x 64   | 1      | float32  |
| gender_googlenet_mnn | 224 x 224 | 3      | float32  |
| example_mnn          | 224 x 224 | 3      | float32  |
| ultraface_detector   | 320 x 240 | 3      | float32  |
| yolov5_detector      | 640 x 640 | 3      | float32  |