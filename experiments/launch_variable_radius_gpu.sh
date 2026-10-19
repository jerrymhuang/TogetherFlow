#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Scope the temporary workaround to this process tree, and only while the old
# kernel module remains loaded. After reboot, use the matching system libraries.
if [[ "$(cat /proc/driver/nvidia/version)" == *580.173.02* ]]; then
    driver_lib=/tmp/togetherflow-nvidia-580.173.02/lib
    if [[ ! -f "$driver_lib/libcuda.so.1" ]]; then
        echo "Matching temporary NVIDIA libraries are missing; restore them or reboot into the installed driver." >&2
        exit 1
    fi
    export LD_LIBRARY_PATH="$driver_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec .venv/bin/python experiments/run_variable_radius_queue.py "$@"
