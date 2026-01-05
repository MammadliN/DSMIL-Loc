#!/bin/bash
# This script installs the Python dependencies needed for DSMIL-Loc and pins
# NumPy to a 1.x release to avoid binary mismatches with wheels built against
# NumPy 1.x.

set -euo pipefail

# Run heavy installs only once per node
DONEFILE="/tmp/dsmil_loc_install_done_${SLURM_JOB_ID:-$$}"

# If this is not the local rank 0, wait for the installation to finish and exit
if [[ "${SLURM_LOCALID:-0}" != "0" ]]; then
  while [[ ! -f "${DONEFILE}" ]]; do
    sleep 1
  done
  exit 0
fi

echo ">>> [install_dsmil] Python version:"
python -c "import sys; print(sys.version)"

echo ">>> [install_dsmil] NumPy version BEFORE install:"
python - << 'PYCHK' || true
try:
    import numpy as np
    print("NumPy:", np.__version__)
except Exception as e:
    print("NumPy not importable:", e)
PYCHK

echo ">>> [install_dsmil] Upgrading pip..."
python -m pip install --upgrade pip --break-system-packages

echo ">>> [install_dsmil] Forcing NumPy 1.26.4 to avoid NumPy 2.x binary issues..."
python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4" --break-system-packages

echo ">>> [install_dsmil] NumPy version AFTER reinstall:"
python - << 'PYCHK'
import numpy as np
print("NumPy:", np.__version__)
PYCHK

echo ">>> [install_dsmil] Removing problematic binary packages (soxr, pyarrow) if present..."
python -m pip uninstall -y soxr pyarrow --break-system-packages || true

echo ">>> [install_dsmil] Installing core scientific/audio stack..."
python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
  "scipy==1.12.0" \
  "pandas==2.2.2" \
  "scikit-learn" \
  "matplotlib==3.9.1" \
  "tqdm" \
  "librosa==0.10.1" \
  "soundfile==0.12.*" \
  "audiomentations==0.36.0" \
  "numba>=0.59,<0.61" \
  --break-system-packages

echo ">>> [install_dsmil] Installing training utilities..."
python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
  "einops==0.8.1" \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  "lightning==2.5.5" \
  "torchmetrics==1.8.2" \
  --break-system-packages

echo ">>> [install_dsmil] Installing torchaudio to match existing torch..."
python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
  torchaudio \
  --break-system-packages

echo ">>> [install_dsmil] Cleaning up unwanted extras..."
python -m pip uninstall -y torch-audiomentations numpy-minmax --break-system-packages || true

echo ">>> [install_dsmil] Final sanity check (torch / torchaudio / numpy):"
python - << 'PYCHK'
import importlib

def safe_import(name):
    try:
        m = importlib.import_module(name)
        print(f"{name}:", getattr(m, "__version__", "no __version__ attr"))
    except Exception as e:
        print(f"{name} import FAILED:", e)

safe_import("numpy")
safe_import("torch")
safe_import("torchaudio")
safe_import("pandas")
PYCHK

touch "${DONEFILE}"
echo ">>> [install_dsmil] Done."
