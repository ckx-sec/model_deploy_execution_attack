#!/bin/bash

# ==============================================================================
# Batch Attack Script for NES Host
#
# This script iterates through all PNG images in a specified source directory
# and runs the nes_attack.py script for each image.
#
# It automatically creates a unique output directory for each attack run
# to prevent results from being overwritten.
# ==============================================================================

# --- Configuration ---
# Base directory for the attack project
BASE_DIR="model_deploy_execution_attack"

# Path to the executable to be attacked
EXECUTABLE="$BASE_DIR/resources/execution_files/mnist_mnn"

# Path to the model file
MODEL="$BASE_DIR/resources/models/mnist.mnn"

# Path to the GDB hooks configuration
HOOKS="$BASE_DIR/hook_config/mnist_mnn_hook_config.json"

# The "golden image" that defines the target state we want to mimic
GOLDEN_IMAGE="$BASE_DIR/resources/images/mnist_sample/7/7_0.png"

# Directory containing the source images to be attacked (we will attack all images in this folder)
SOURCE_IMAGE_DIR="$BASE_DIR/resources/images/mnist_sample/0"

# Parent directory for all attack outputs
BASE_OUTPUT_DIR="$BASE_DIR/outputs/mnist_results"

# Check if the source image directory exists
if [ ! -d "$SOURCE_IMAGE_DIR" ]; then
    echo "Error: Source image directory not found at '$SOURCE_IMAGE_DIR'"
    exit 1
fi

# Loop through all .png files in the source directory
for image_path in "$SOURCE_IMAGE_DIR"/*.png; do
    # This check handles the case where the directory is empty or contains no png files
    [ -e "$image_path" ] || continue

    echo "===================================================="
    echo "Starting attack on: $image_path"
    echo "===================================================="

    # --- Create a unique output directory for this run ---
    # Extract the base filename (e.g., "0_0.png")
    image_filename=$(basename "$image_path")
    # Remove the file extension (e.g., "0_0")
    image_name_no_ext="${image_filename%.*}"
    # Create a descriptive and unique directory name for the output
    specific_output_dir="$BASE_OUTPUT_DIR/${image_name_no_ext}_attacked_to_be_7"
    
    # Create the directory if it doesn't exist
    mkdir -p "$specific_output_dir"


    # --- Execute the Attack Command ---
    python3 "$BASE_DIR/src/attackers/nes_attack.py" \
        --executable "$EXECUTABLE" \
        --model "$MODEL" \
        --hooks "$HOOKS" \
        --golden-image "$GOLDEN_IMAGE" \
        --image "$image_path" \
        --output-dir "$specific_output_dir" \
        --iterations 500 \
        --learning-rate 5.0 \
        --population-size 200 \
        --sigma 0.2 \
        --workers 32 \
        --enable-warm-restarts \
        --lr-restart-cycles 5 \
        --lr-restart-cycle-len 50 \
        --lr-restart-cycle-mult 1

    echo ""
    echo "Attack on $image_path finished."
    echo "Results saved in: $specific_output_dir"
    echo "===================================================="
    echo ""
done

echo "All attacks completed." 