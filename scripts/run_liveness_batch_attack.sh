#!/bin/bash

# ==============================================================================
# Batch Attack Script for Liveness Model
#
# This script reads a list of image filenames from a text file, constructs
# the full image paths, and runs an attack script for each image.
#
# It automatically creates a unique output directory for each attack run.
# ==============================================================================

# --- Configuration ---
# !!! 请在下方填入确切的配置信息 !!!

# 攻击项目的根目录
BASE_DIR="/home/ckx/model_deploy_execution_attack"

# 需要攻击的可执行文件路径
EXECUTABLE="$BASE_DIR/resources/execution_files/liveness_example"

# MODEL文件路径 (例如: "$BASE_DIR/resources/models/liveness.mnn")
# !!! 请提供模型文件的路径 !!!
MODEL="/home/ckx/insightface/cpp-package/inspireface/test_res/pack/Pikachu"

# GDB钩子配置文件路径 (例如: "$BASE_DIR/hook_config/liveness_hooks.json")
# !!! 请提供GDB钩子配置文件的路径 !!!
HOOKS="$BASE_DIR/hook_config/liveness_example_hook_config.json"

# “黄金图片”的路径，该图片定义了我们希望模仿的目标状态 (例如: 一张被正确识别为“活体”的人脸图片)
# !!! 请提供黄金图片的路径 !!!
GOLDEN_IMAGE="/home/ckx/img_align_celeba/000018.jpg"

# 包含待攻击图片文件名列表的文本文件
SPOOF_FACES_LIST="$BASE_DIR/resources/list/spoof_faces.txt"

# 待攻击图片所在的基础目录
IMAGE_BASE_PATH="/home/ckx/img_align_celeba"

# 所有攻击结果的父输出目录
BASE_OUTPUT_DIR="$BASE_DIR/outputs/liveness_results"

# 使用的攻击脚本 (以NES为例，你可以根据需要更改)
ATTACK_SCRIPT="$BASE_DIR/src/attackers/nes_attack.py"

# --- Sanity Checks ---
if [ ! -f "$MODEL" ]; then
    echo "错误: 模型文件不存在于 '$MODEL'"
    exit 1
fi

if [ ! -f "$HOOKS" ]; then
    echo "错误: GDB钩子配置文件不存在于 '$HOOKS'"
    exit 1
fi

if [ ! -f "$GOLDEN_IMAGE" ]; then
    echo "错误: 黄金图片不存在于 '$GOLDEN_IMAGE'"
    exit 1
fi

if [ ! -f "$SPOOF_FACES_LIST" ]; then
    echo "错误: 未在 '$SPOOF_FACES_LIST' 找到图片列表文件"
    exit 1
fi

if [ ! -d "$IMAGE_BASE_PATH" ]; then
    echo "错误: 未在 '$IMAGE_BASE_PATH' 找到图片基础目录"
    exit 1
fi

# --- Main Loop ---

# 如果基础输出目录不存在，则创建它
mkdir -p "$BASE_OUTPUT_DIR"

# 逐行读取图片列表文件
while IFS= read -r image_filename || [[ -n "$image_filename" ]]; do
    # 过滤掉空行
    if [ -z "$image_filename" ]; then
        continue
    fi

    # 构建源图片的完整路径
    image_path="$IMAGE_BASE_PATH/$image_filename"

    # 检查源图片文件是否存在
    if [ ! -f "$image_path" ]; then
        echo "警告: 源图片不存在，跳过: $image_path"
        continue
    fi

    echo "===================================================="
    echo "开始攻击: $image_path"
    echo "===================================================="

    # --- 为本次运行创建唯一的输出目录 ---
    # 提取基础文件名 (例如 "000001.jpg")
    base_filename=$(basename "$image_path")
    # 移除文件扩展名 (例如 "000001")
    image_name_no_ext="${base_filename%.*}"
    # 为输出创建一个描述性的、唯一的目录名
    specific_output_dir="$BASE_OUTPUT_DIR/${image_name_no_ext}_attacked_to_be_live"
    
    # 创建目录
    mkdir -p "$specific_output_dir"

    # --- 执行攻击命令 ---
    # 以NES攻击为例，你可能需要调整攻击脚本及其超参数
    python3 "$ATTACK_SCRIPT" \
        --executable "$EXECUTABLE" \
        --model "$MODEL" \
        --hooks "$HOOKS" \
        --golden-image "$GOLDEN_IMAGE" \
        --image "$image_path" \
        --output-dir "$specific_output_dir" \
        --iterations 200 \
        --learning-rate 5.0 \
        --lr-decay-rate 0.97 \
        --lr-decay-steps 50 \
        --population-size 200 \
        --sigma 0.2 \
        --workers 32 \
        --enable-stagnation-decay \
        --stagnation-patience 10

    echo ""
    echo "对 $image_path 的攻击已完成。"
    echo "结果保存在: $specific_output_dir"
    echo "===================================================="
    echo ""
done < <(tac "$SPOOF_FACES_LIST")

echo "所有攻击任务已完成。" 