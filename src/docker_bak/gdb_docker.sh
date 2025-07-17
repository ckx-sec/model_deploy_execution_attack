#!/bin/bash

# A simple wrapper to execute GDB inside a Docker container.
# It assumes all necessary files (executable, models, scripts) have already
# been copied to the correct locations inside the container's workdir.

# --- Configuration ---
CONTAINER_NAME="greybox_attacker_container"
REMOTE_WORKDIR="/app"

# The command to execute, where all paths are relative to the container.
CMD_TO_RUN=("$@")

# --- Validation ---
if [ "$#" -lt 1 ]; then
    # Output to stderr
    echo "Usage: $0 <path_to_executable_in_container> [arg1] [arg2]..." >&2
    exit 1
fi

# --- Execution ---
# Execute GDB in batch mode. The Python orchestrator script is responsible
# for ensuring gdb_script.py and all program arguments are in place.
docker exec \
  -w "${REMOTE_WORKDIR}" \
  -e "LD_LIBRARY_PATH=${REMOTE_WORKDIR}/third_party/mnn/lib:${REMOTE_WORKDIR}/third_party/onnxruntime/lib" \
  "${CONTAINER_NAME}" \
  gdb -batch -x "${REMOTE_WORKDIR}/gdb_script.py" --args "${CMD_TO_RUN[@]}" 