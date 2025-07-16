# Model Deploy Execution Attack Lab

本项目是一个面向**AI模型二进制部署安全**的自动化对抗攻击实验平台，支持多种主流推理引擎（MNN、NCNN、ONNXRuntime等）和多种攻击算法。核心特性包括：
- 支持**白盒（状态匹配）**与**灰盒（决策型）**攻击
- 通过GDB hook自动提取模型内部状态，无需源码
- 自动适配图片通道数/尺寸，兼容彩色与灰度模型
- 支持多进程并行、内存优化、详细日志输出

## 支持的攻击算法
- **CMA-ES**：白盒，适合低/中维度输入，能高效拟合目标内部状态
- **NES**：白盒，适合高维输入，内存消耗低，梯度估计高效
- **Boundary Attack**、**HopSkipJump**、**Sign-OPT**：灰盒，适合只关心最终决策的场景

## 典型用法（以ssrnet_age_mnn为例）

### NES攻击（白盒，状态匹配）
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

### CMA-ES攻击（白盒，状态匹配）
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





目前的示例代码

测试结果，

污点分析

对其中loss function的设计做修改current_hooks和target_hooks传入时要附带上偏移地址的信息，目前类似于Target hooks captured: [-9.59821033, 11.9071264, 11.9071264, 15.8226147, 15.8226147, 7.97491789, 15.8226147, -10.4998245, 15.8226147, -15.2762852, 15.8226147, -19.3611851, 15.8226147, 21.8360138, 21.8360138, -5.58873224, -3.16794729, 21.8360138, 4.86246645e-05, 2.22437459e-14, 4.86246645e-05, 0.00243967259, 9.53060237e-07, 0.00243967259, 0.00243967259, 9.02903749e-15, 7.60767208e-17, 0.00243967259, 0.00243967259, 1.27997742e-18, 0.997510791, 0.00243967259, 1.2260324e-12, 0.997510791, 1.37986012e-11, 0.997510791, 0.997510791, 0.800000012]，要修改成{"0x.....":[-9.59821033, 11.9071264],"0x....":[11.9071264, 15.8226147]，当current_hooks和target_hooks有一些"0x....."字段缺失，则只考虑共同存在的"0x....."字段，然后对应做mse
