#!/bin/bash

# A simple wrapper to execute GDB on the host machine for the attack script.
# It sets up library paths and executes GDB with the correct arguments.

# --- Validation ---
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <abs_path_to_executable> <abs_path_to_model> <abs_path_to_image>" >&2
    exit 1
fi

# --- Path Setup ---
# Get the absolute path to the directory containing this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="${SCRIPT_DIR}/../.." # Go up to the project root from src/attackers

# Define paths to the required libraries and scripts relative to the project root.
MNN_LIB_PATH="${PROJECT_ROOT}/third_party/mnn/lib"
ONNX_LIB_PATH="${PROJECT_ROOT}/third_party/onnxruntime/lib"
GDB_SCRIPT_PATH="${SCRIPT_DIR}/gdb_script_host.py" # Use the new host-specific GDB script

# The executable and its arguments are passed from the Python script.
# The hooks file path is the last argument.
EXECUTABLE_PATH="$1"
MODEL_PATH="$2"
IMAGE_PATH="$3"
# Export the path to the hooks file as an environment variable so the GDB script can read it reliably.
export HOOKS_JSON_PATH="$4" 

# --- Pre-run Checks ---
if [ ! -f "${GDB_SCRIPT_PATH}" ]; then
    echo "Error: GDB script not found at ${GDB_SCRIPT_PATH}" >&2
    exit 1
fi
for lib_path in "${MNN_LIB_PATH}" "${ONNX_LIB_PATH}"; do
    if [ ! -d "$lib_path" ]; then
        echo "Warning: Library directory not found: $lib_path" >&2
    fi
done


# --- Execution ---
# Set the LD_LIBRARY_PATH to include our custom-built libraries so the executable can find them.
# Then, execute GDB in batch mode, running the Python gdb script.
# The --args flag passes all subsequent arguments to the program being debugged.
echo "Running GDB with script: ${GDB_SCRIPT_PATH}"
echo "Executable and args: ${EXECUTABLE_PATH} ${MODEL_PATH} ${IMAGE_PATH}"
echo "Using hooks file from env: ${HOOKS_JSON_PATH}"
echo "Using LD_LIBRARY_PATH: ${MNN_LIB_PATH}:${ONNX_LIB_PATH}"

LD_LIBRARY_PATH="${MNN_LIB_PATH}:${ONNX_LIB_PATH}:${LD_LIBRARY_PATH}" \
  gdb -batch -x "${GDB_SCRIPT_PATH}" --args "${EXECUTABLE_PATH}" "${MODEL_PATH}" "${IMAGE_PATH}" 