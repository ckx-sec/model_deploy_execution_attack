#!/bin/bash

# This script processes images using different models and classifies them based on the model's output.

set -e

# Define base directory relative to the script location
# This makes the script runnable from any directory
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
BASE_DIR=$(realpath "$SCRIPT_DIR/..")

# Set LD_LIBRARY_PATH to include third party libraries
export LD_LIBRARY_PATH=$BASE_DIR/third_party/mnn/lib:$BASE_DIR/third_party/ncnn/lib:$BASE_DIR/third_party/onnxruntime/lib:$BASE_DIR/third_party/pplnn/lib:$BASE_DIR/third_party/tnn/lib:$LD_LIBRARY_PATH


# Define directories
EXEC_DIR="$BASE_DIR/resources/execution_files"
DEFAULT_IMAGE_DIR="$BASE_DIR/resources/images/img_align_celeba"
MODEL_DIR="$BASE_DIR/resources/models" # Corrected model directory path
RESULT_DIR="$BASE_DIR/results"

# Check if required directories exist
if [ ! -d "$EXEC_DIR" ]; then
  echo "Error: Executables directory not found at $EXEC_DIR"
  exit 1
fi
if [ ! -d "$DEFAULT_IMAGE_DIR" ]; then
  echo "Error: Images directory not found at $DEFAULT_IMAGE_DIR"
  exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: Models directory not found at $MODEL_DIR"
  exit 1
fi

# Create results directory
mkdir -p "$RESULT_DIR"
echo "Results will be stored in $RESULT_DIR"

# Loop through each executable in the execution_files directory
for exec_path in "$EXEC_DIR"/*; do
  if [ ! -f "$exec_path" ] || [ ! -x "$exec_path" ]; then
    echo "Skipping non-executable file: $exec_path"
    continue
  fi
  
  exec_name=$(basename "$exec_path")
  echo "--- Processing with executable: $exec_name ---"
  
  # Determine model name and framework from executable name
  exec_base_name=""
  model_base_name=""
  model_base_name2=""
  model_file_path=""
  model_file_path2=""
  ncnn_bin_path=""

  # Determine framework and base executable name
  if [[ "$exec_name" == *"_mnn" ]]; then
    exec_base_name="${exec_name%_mnn}"
  elif [[ "$exec_name" == *"_ncnn" ]]; then
    exec_base_name="${exec_name%_ncnn}"
  elif [[ "$exec_name" == *"_onnxruntime" ]]; then
    exec_base_name="${exec_name%_onnxruntime}"
  elif [[ "$exec_name" == *"_tnn" ]]; then
    exec_base_name="${exec_name%_tnn}"
  else
    echo "Skipping $exec_name: Could not determine model framework."
    continue
  fi

  # Select image directory based on model
  IMAGE_DIR="$DEFAULT_IMAGE_DIR"
  if [ "$exec_base_name" == "mnist" ]; then
    IMAGE_DIR="$BASE_DIR/resources/images/mnist_sample"
    if [ ! -d "$IMAGE_DIR" ]; then
      echo "Error: MNIST image directory not found at $IMAGE_DIR"
      continue
    fi
  fi

  # Map executable name to model name
  case "$exec_base_name" in
    "fsanet_headpose")
      model_base_name="fsanet-var"
      ;;
    "emotion_ferplus")
      model_base_name="emotion-ferplus-8"
      ;;
    "pfld_landmarks")
      model_base_name="pfld-106-lite"
      ;;
    *)
      model_base_name="$exec_base_name"
      ;;
  esac

  # Construct model file paths based on framework
  if [[ "$exec_name" == *"_mnn" ]]; then
    model_file_path="$MODEL_DIR/$model_base_name.mnn"
    if [[ -n "$model_base_name2" ]]; then
      model_file_path2="$MODEL_DIR/$model_base_name2.mnn"
    fi
  elif [[ "$exec_name" == *"_ncnn" ]]; then
    model_file_path="$MODEL_DIR/$model_base_name.param"
    ncnn_bin_path="$MODEL_DIR/$model_base_name.bin"
    if [ ! -f "$ncnn_bin_path" ]; then
        echo "Error: .bin file for NCNN model $model_base_name not found at $ncnn_bin_path. Skipping."
        continue
    fi
  elif [[ "$exec_name" == *"_onnxruntime" ]]; then
    model_file_path="$MODEL_DIR/$model_base_name.onnx"
    if [[ -n "$model_base_name2" ]]; then
      model_file_path2="$MODEL_DIR/$model_base_name2.onnx"
    fi
  elif [[ "$exec_name" == *"_tnn" ]]; then
    model_file_path="$MODEL_DIR/$model_base_name.tnnproto"
     if [ ! -f "$MODEL_DIR/$model_base_name.tnnmodel" ]; then
        echo "Warning: .tnnmodel file for TNN model $model_base_name not found. The executable might fail."
    fi
  fi

  if [ ! -f "$model_file_path" ]; then
    echo "Model file not found for $exec_name at $model_file_path. Skipping."
    continue
  fi

  if [[ -n "$model_file_path2" && ! -f "$model_file_path2" ]]; then
    echo "Second model file not found for $exec_name at $model_file_path2. Skipping."
    continue
  fi

  # Create output directories for the current model
  output_dir_true="$RESULT_DIR/$exec_name/true"
  output_dir_false="$RESULT_DIR/$exec_name/false"
  mkdir -p "$output_dir_true"
  mkdir -p "$output_dir_false"

  echo "Processing images from $IMAGE_DIR..."
  # Loop through each image
  find "$IMAGE_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | while read -r image_path; do
    if [ ! -f "$image_path" ]; then
      continue
    fi

    image_name=$(basename "$image_path")
    echo "  - Analyzing $image_name..."

    # Run the executable with the model and image, and capture the output
    # And it prints "true" or "false" to stdout.
    output=""
    exit_code=0
    if [[ -n "$model_file_path2" ]]; then
      # Case for executables requiring two models (e.g., fsanet_headpose)
      output=$("$exec_path" "$model_file_path" "$model_file_path2" "$image_path" 2>&1) || exit_code=$?
    elif [[ "$exec_name" == *"_ncnn" ]]; then
      # NCNN requires .param and .bin files for a single model
      output=$("$exec_path" "$model_file_path" "$ncnn_bin_path" "$image_path" 2>&1) || exit_code=$?
    else
      # Default case for single-model executables
      output=$("$exec_path" "$model_file_path" "$image_path" 2>&1) || exit_code=$?
    fi

    if [ $exit_code -ne 0 ]; then
      echo "    -> Error processing $image_name with $exec_name (exit code $exit_code). Skipping."
      echo "    -> Error details: $output"
      continue
    fi

    # Classify the image based on the result, trimming whitespace from output
    result=$(echo "$output" | tr -d '[:space:]')
    if [[ "$result" == *"true"* ]]; then
      cp "$image_path" "$output_dir_true/"
      echo "    -> Result: true. Copied to $output_dir_true"
    else
      cp "$image_path" "$output_dir_false/"
      echo "    -> Result: false. Copied to $output_dir_false"
    fi
  done

  echo "--- Finished processing for $exec_name ---"
  echo ""
done

echo "All processing complete." 