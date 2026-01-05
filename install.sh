#!/bin/bash
# This script installs the Python dependencies needed for DSMIL-Loc.

set -euo pipefail

echo ">>> [install_dsmil] Python version:"
python -c "import sys; print(sys.version)"

echo ">>> [install_dsmil] NumPy version BEFORE install:"
python -c "import numpy; print(numpy.__version__)" || echo "NumPy not installed yet"

# 1) Make sure pip is recent
python -m pip install --upgrade pip

# 2) Force NumPy to a 1.x version that is known to be stable
echo ">>> [install_dsmil] Installing NumPy 1.26.4..."
python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

echo ">>> [install_dsmil] NumPy version AFTER numpy install:"
python -c "import numpy; print(numpy.__version__)"

# 3) Remove soxr completely; librosa can work without it.
echo ">>> [install_dsmil] Uninstalling soxr (if present)..."
python -m pip uninstall -y soxr || true

# 4) Install core scientific and audio libraries.
echo ">>> [install_dsmil] Installing librosa and audio stack..."
python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
    "librosa==0.10.1" \
    "soundfile==0.12.*" \
    scipy \
    pandas \
    scikit-learn \
    matplotlib \
    tqdm \
    "numba>=0.59,<0.61"

# 5) Training/utility libraries
#    We avoid forcing any upgrade that would bump NumPy again.
echo ">>> [install_dsmil] Installing training utilities..."
python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
    lightning \
    torchmetrics \
    einops \
    hydra-core \
    audiomentations

# 6) Project extras (match DSMIL-Loc requirements without re-upgrading numpy)
echo ">>> [install_dsmil] Installing project requirements..."
python -m pip install --no-cache-dir --upgrade-strategy only-if-needed -r requirements.txt

# 7) Remove packages that were explicitly uninstalled locally
echo ">>> [install_dsmil] Cleaning up unwanted packages..."
python -m pip uninstall -y torch-audiomentations numpy-minmax || true

# 8) Final check
echo ">>> [install_dsmil] FINAL NumPy version:"
python -c "import numpy; print(numpy.__version__)"
