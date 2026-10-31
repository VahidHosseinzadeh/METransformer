#!/bin/bash
# Bridges-2: PSC's PyTorch module (torch, torchvision, numpy, matplotlib) plus a thin venv for
# the two packages the module lacks, wandb and pytest.   bash scripts/setup_bridges2.sh
# The module's environment is itself a (uv) venv, so --system-site-packages would expose only
# its base interpreter; a .pth file adds the module's site-packages to the thin venv instead,
# after the venv's own. pip then sees the module's packages and installs only what is missing.
cd "$(dirname "$0")/.."
MODULE=${MODULE:-pytorch/26.05-2.11-py3}
source /etc/profile > /dev/null 2>&1  # defines `module`; its exit status means nothing here
module load "$MODULE" > /dev/null 2>&1
[ -x "$ENV/bin/python" ] || { echo "module $MODULE did not provide \$ENV/bin/python"; exit 1; }
set -eo pipefail
rm -rf .venv
"$ENV/bin/python" -m venv .venv
purelib() { "$1" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"; }
echo "import site; site.addsitedir('$(purelib "$ENV/bin/python")')" \
    > "$(purelib .venv/bin/python)/psc_pytorch_module.pth"
.venv/bin/pip install -q --no-cache-dir wandb pytest
.venv/bin/python - <<'PY'
import matplotlib, numpy, pytest, sys, torch, torchvision, wandb
print("python", sys.version.split()[0], "torch", torch.__version__, "cudnn",
      torch.backends.cudnn.version(), "torchvision", torchvision.__version__, "numpy",
      numpy.__version__, "wandb", wandb.__version__, "pytest", pytest.__version__)
print("torch from", torch.__file__)
PY
