# Pi05 TensorRT — Environment Setup & Model Conversion

This document describes how to set up the environment for **Pi05 TensorRT** on Jetson and convert trained Pi05 models through the following pipeline:

```text
JAX checkpoint
    ↓
PyTorch
    ↓
ONNX (FP16 / FP8)
    ↓
TensorRT Engine
```

Repository:

```text
https://github.com/Hung6779/Pi05_Tensorrt_Release.git
```

---

## 1. Environment Setup

### 1.1 Clone the repository

```bash
git clone https://github.com/Hung6779/Pi05_Tensorrt_Release.git
cd Pi05_Tensorrt_Release
```

### 1.2 Create Python 3.10 virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Set the Python site-packages path:

```bash
VENV_SITE=.venv/lib/python3.10/site-packages
```

> **Note:** The project currently uses Python 3.10 for the TensorRT deployment environment.

---

## 2. Install PyTorch for Jetson

Install the Jetson-compatible PyTorch and torchvision packages from the NVIDIA Jetson AI Lab repository:

```bash
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  torch==2.8.0 torchvision==0.23.0
```

Allow the virtual environment to access system Python packages:

```bash
echo "/usr/lib/python3.10/dist-packages" > "$VENV_SITE/_system_dist_packages.pth"
```

---

## 3. Install Required Dependencies

### 3.1 Basic dependencies

```bash
pip install fsspec gcsfs tqdm-loggable
```

### 3.2 OpenPI / model dependencies

```bash
pip install safetensors tyro einops beartype jaxtyping augmax dm-tree equinox \
  flatbuffers gym-aloha imageio ml-collections numpydantic opencv-python pillow \
  sentencepiece wandb chex transformers polars "orbax-checkpoint==0.11.13" \
  "numpy>=1.26,<2.0.0"
```

### 3.3 Deployment and dataset dependencies

```bash
pip install "datasets==3.6.0" av deepdiff diffusers draccus flask gdown \
  "gymnasium==0.29.1" h5py jsonlines numba omegaconf opencv-python-headless \
  pymunk pynput pyzmq rerun-sdk termcolor "zarr>=2.17.0,<3.0.0"
```

> **Correction:** If `deepdeep` above is not available in your environment, use the intended package:
>
> ```bash
> pip install "datasets==3.6.0" av deepdiff diffusers draccus flask gdown \
>   "gymnasium==0.29.1" h5py jsonlines numba omegaconf opencv-python-headless \
>   pymunk pynput pyzmq rerun-sdk termcolor "zarr>=2.17.0,<3.0.0"
> ```

---

## 4. Install LeRobot

Install the pinned LeRobot revision:

```bash
pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
```

---

## 5. Install OpenPI

Install the OpenPI client:

```bash
pip install -e packages/openpi-client
```

Install the project itself.

The project `pyproject.toml` requires Python >= 3.11, while this deployment environment uses Python 3.10. Therefore, bypass the Python version requirement:

```bash
pip install --ignore-requires-python -e .
```

---

## 6. Patch Transformers

Pi05 uses custom Transformer implementations. Copy the replacement modules into the installed `transformers` package:

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
  "$VENV_SITE/transformers/"
```

If the shell reports that the target directory does not exist, verify the installation first:

```bash
python -c "import transformers; print(transformers.__path__[0])"
```

---

## 7. Install ONNX / Model Optimization Dependencies

Install NVIDIA ModelOpt and ONNX tools:

```bash
pip install "nvidia-modelopt==0.33.1" onnx onnxslim lief
```

Install the Jetson-compatible ONNX Runtime GPU package:

```bash
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  onnxruntime-gpu==1.24.0
```

---

## 8. LeRobot Dataset Compatibility Patch

Create the expected dataset package path:

```bash
mkdir -p "$VENV_SITE/lerobot/datasets"
```

Create the package initializer:

```bash
touch "$VENV_SITE/lerobot/datasets/__init__.py"
```

Create the compatibility wrapper:

```bash
cat > "$VENV_SITE/lerobot/datasets/lerobot_dataset.py" <<'EOF'
from lerobot.common.datasets.lerobot_dataset import *  # noqa: F401,F403
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: F401
EOF
```

---

# 9. Convert Trained JAX Model to PyTorch

The conversion pipeline downloads the trained JAX checkpoint and converts it to PyTorch.

## 9.1 Select the model configuration

```bash
export CONFIG_NAME=pi05_libero
```

Download the checkpoint:

```bash
python -c "
from openpi.shared import download
print(download.maybe_download(f'gs://openpi-assets/checkpoints/{CONFIG_NAME}'))
"
```

For the example above, the checkpoint is expected at:

```text
/home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero
```

---

## 9.2 Apply Transformer replacement

If the conversion environment is a different virtual environment, copy the Transformer replacements into that environment:

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
  /home/hung/hungvd27/Pi05_ONNX_Tensorrt/.venv/lib/python3.10/site-packages/transformers/
```

> Update the path above if your repository or virtual environment is located elsewhere.

---

## 9.3 Convert JAX checkpoint to PyTorch

Run:

```bash
python examples/convert_jax_model_to_pytorch.py \
  --config-name pi05_libero \
  --checkpoint-dir /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --output-path /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

The output directory will contain the converted PyTorch checkpoint:

```text
/home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

---

# 10. Convert PyTorch to ONNX

The PyTorch checkpoint can be exported to ONNX using the deployment conversion script.

## 10.1 FP16 ONNX

```bash
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --output_path /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --config_name pi05_libero \
  --precision fp16
```

This generates the FP16 ONNX model.

---

## 10.2 FP8 ONNX

For FP8 conversion:

```bash
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --output_path /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --config_name pi05_libero \
  --precision fp8 \
  --quantize_attention_matmul
```

The `--quantize_attention_matmul` option enables attention matrix multiplication quantization during the FP8 conversion flow.

---

# 11. Convert ONNX to TensorRT

After generating the ONNX model, build the TensorRT engine using:

```bash
ulimit -c 0

MAX_BATCH=1 OPT_BATCH=1 MIN_BATCH=1 \
  ./deployment_scripts/build_engine.sh \
  <path/to/model.onnx> \
  <path/to/output.engine>
```

For example:

```bash
ulimit -c 0

MAX_BATCH=1 OPT_BATCH=1 MIN_BATCH=1 \
  ./deployment_scripts/build_engine.sh \
  /path/to/pi05_model.onnx \
  /path/to/pi05_model.engine
```

The resulting `.engine` file can then be used for TensorRT inference.

---

# 12. Complete Conversion Flow

The complete workflow is:

```text
┌──────────────────────┐
│  JAX Checkpoint      │
│  pi05_libero         │
└──────────┬───────────┘
           │
           │ convert_jax_model_to_pytorch.py
           ▼
┌──────────────────────┐
│  PyTorch Checkpoint  │
│  pi05_libero         │
└──────────┬───────────┘
           │
           │ pytorch_to_onnx.py
           ▼
┌──────────────────────┐
│      ONNX Model      │
│   FP16 / FP8         │
└──────────┬───────────┘
           │
           │ build_engine.sh
           ▼
┌──────────────────────┐
│   TensorRT Engine    │
│       *.engine       │
└──────────────────────┘
```

---

# 13. Quick Setup — All Commands

For a clean environment, the main setup commands are:

```bash
git clone https://github.com/Hung6779/Pi05_Tensorrt_Release.git
cd Pi05_Tensorrt_Release

python3.10 -m venv .venv
source .venv/bin/activate

VENV_SITE=.venv/lib/python3.10/site-packages

pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  torch==2.8.0 torchvision==0.23.0

echo "/usr/lib/python3.10/dist-packages" > "$VENV_SITE/_system_dist_packages.pth"

pip install fsspec gcsfs tqdm-loggable

pip install safetensors tyro einops beartype jaxtyping augmax dm-tree equinox \
  flatbuffers gym-aloha imageio ml-collections numpydantic opencv-python pillow \
  sentencepiece wandb chex transformers polars "orbax-checkpoint==0.11.13" \
  "numpy>=1.26,<2.0.0"

pip install "datasets==3.6.0" av deepdiff diffusers draccus flask gdown \
  "gymnasium==0.29.1" h5py jsonlines numba omegaconf opencv-python-headless \
  pymunk pynput pyzmq rerun-sdk termcolor "zarr>=2.17.0,<3.0.0"

pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"

pip install -e packages/openpi-client
pip install --ignore-requires-python -e . # pip install --ignore-requires-python -e . --no-deps

cp -r ./src/openpi/models_pytorch/transformers_replace/* \
  "$VENV_SITE/transformers/"

pip install "nvidia-modelopt==0.33.1" onnx onnxslim lief

pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  onnxruntime-gpu==1.24.0

mkdir -p "$VENV_SITE/lerobot/datasets"
touch "$VENV_SITE/lerobot/datasets/__init__.py"

cat > "$VENV_SITE/lerobot/datasets/lerobot_dataset.py" <<'EOF'
from lerobot.common.datasets.lerobot_dataset import *  # noqa: F401,F403
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: F401
EOF
```

---

# 14. Quick Conversion Commands

### JAX → PyTorch

```bash
export CONFIG_NAME=pi05_libero

python -c "
from openpi.shared import download
print(download.maybe_download(f'gs://openpi-assets/checkpoints/{CONFIG_NAME}'))
"

python examples/convert_jax_model_to_pytorch.py \
  --config-name pi05_libero \
  --checkpoint-dir /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --output-path /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

### PyTorch → ONNX FP16

```bash
cp /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/assets/physical-intelligence/libero/

python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --output_path /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --config_name pi05_libero \
  --precision fp16
```

### PyTorch → ONNX FP8

```bash
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --output_path /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --config_name pi05_libero \
  --precision fp8 \
  --quantize_attention_matmul
```

### ONNX → TensorRT

```bash
ulimit -c 0

MAX_BATCH=1 OPT_BATCH=1 MIN_BATCH=1 \
  ./deployment_scripts/build_engine.sh \
  <path/to/model.onnx> \
  <path/to/output.engine>
```

---

# 15. Environment Verification

After installation, verify the main packages:

```bash
python --version
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torchvision; print('TorchVision:', torchvision.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import onnx; print('ONNX:', onnx.__version__)"
python -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"
```

Check CUDA availability from PyTorch:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda)"
```

Check TensorRT:

```bash
python -c "import tensorrt as trt; print('TensorRT:', trt.__version__)"
```

---

# 16. Notes

- This setup is intended for a **Jetson / NVIDIA JetPack 6 + CUDA 12.6** environment.
- Python 3.10 is used for the deployment environment.
- The project installation uses `--ignore-requires-python` because the current `pyproject.toml` requires Python >= 3.11.
- Keep the pinned LeRobot commit unchanged unless there is a specific reason to update it.
- The `transformers_replace` files are required by the Pi05 PyTorch conversion/deployment flow.
- Use `FP16` when you need a straightforward reduced-precision ONNX model.
- Use `FP8` with `--quantize_attention_matmul` when the target TensorRT / Jetson software stack supports the required FP8 flow.
- TensorRT engine files are hardware/software-stack dependent. Build the engine on the target Jetson environment whenever possible.

---

## Repository

Pi05 TensorRT Release:

```text
https://github.com/Hung6779/Pi05_Tensorrt_Release
```
