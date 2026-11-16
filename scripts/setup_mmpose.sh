#!/usr/bin/env bash
# Install MMPose stack into the active conda env (torch 2.1.2 + CUDA 12.1).
# mmcv must match torch/CUDA exactly; mmpose is installed with --no-deps because
# its 'chumpy' dependency (SMPL only, unused here) fails to build.
set -euo pipefail

pip install "setuptools<81"  # mmengine imports pkg_resources
pip install "mmengine==0.10.7"
pip install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install "mmdet==3.3.0"  # imported by mmpose 1.3.2 heads
pip install --no-deps "mmpose==1.3.2"
pip install json_tricks munkres scipy matplotlib pillow xtcocotools "numpy<2"

python -c "import mmcv, mmpose, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| mmcv', mmcv.__version__, '| mmpose', mmpose.__version__)"
