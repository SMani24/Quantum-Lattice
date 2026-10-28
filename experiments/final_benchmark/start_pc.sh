#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
task_python="${task_root}/.venv/bin/python"
if [[ ! -x "$task_python" ]]; then
  echo "The shared research .venv is missing. Use the staged PC checkout with its verified environment."
  exit 2
fi
if systemctl --user is-active --quiet quantum-lattice-final.service; then
  echo "The final campaign is already running. Use run.py --status or --pause."
  exit 0
fi
cd -- "$task_root"
exec systemd-run --user --unit=quantum-lattice-final --collect \
  --property=Restart=no --property=KillMode=mixed --property=KillSignal=SIGINT \
  --property=TimeoutStopSec=330 \
  --working-directory="$task_root" \
  "$task_python" experiments/final_benchmark/run.py --run --resume "$@"
