# Convert Pi0.5 (openpi) sang PyTorch / ONNX / TensorRT trên Jetson AGX Orin

Tài liệu này ghi lại toàn bộ quy trình convert checkpoint `pi05_libero` từ JAX sang
PyTorch, ONNX rồi TensorRT trên **Jetson AGX Orin** (JetPack 6.4.3 / L4T R36.4.3,
CUDA 12.6, TensorRT 10.3.0), cùng các bug gặp phải và cách fix.

> Lưu ý quan trọng: `deployment_scripts/` (bao gồm `thor.Dockerfile`,
> `build_engine.sh` gốc, pipeline FP8/NVFP4) được thiết kế cho **Jetson Thor**
> (JetPack 7.2, CUDA 13.0+, GPU kiến trúc mới có FP8 Tensor Core). Trên Orin,
> một số bước cần điều chỉnh như mô tả bên dưới.

## Kết quả cuối cùng trên Jetson AGX Orin

| Bước | Kết quả |
|---|---|
| JAX → PyTorch | ✅ Thành công |
| PyTorch → ONNX (FP16) | ✅ Thành công |
| **ONNX → TensorRT engine (FP16)** | **✅ PASSED — dùng thực tế được** (`model_fp16_seq200.engine`, ~6.1GB, ~300ms/inference) |
| PyTorch → ONNX (FP8 + quantize_attention_matmul) | ✅ Export ra file ONNX thành công |
| ONNX → TensorRT engine (FP8) | ❌ **Không build được — giới hạn phần cứng** (Orin/Ampere không có FP8 Tensor Core, xem Bug 6). Không phải lỗi code, không sửa được trên máy này. |

**→ Trên AGX Orin, FP16 là lựa chọn engine dùng để triển khai thực tế.**

## Môi trường

- Venv: `/home/hung/hungvd27/Pi05_ONNX_Tensorrt/.venv` (Python 3.10.12 — lưu ý
  `pyproject.toml` chính của openpi yêu cầu `>=3.11`, nhưng venv hiện tại là 3.10;
  cài đặt dùng `pip install --ignore-requires-python` khi cần).
- Torch: cài từ index riêng cho Jetson —
  `pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch==2.8.0 torchvision==0.23.0`
  (bản CUDA thật, khác với torch CPU-only lấy từ PyPI thường).
- TensorRT: dùng bản hệ thống cài qua apt (`python3-libnvinfer`, version 10.3.0),
  venv truy cập qua file `.pth` trỏ tới `/usr/lib/python3.10/dist-packages`.
- `nvidia-modelopt`: **phải là `0.33.1`**, không dùng bản mới nhất (xem Bug 4).

## Cài đặt môi trường từ đầu (A → Z)

Giả định: đã có venv Python 3.10 tại `/home/hung/hungvd27/Pi05_ONNX_Tensorrt/.venv`
và đã activate (`source .venv/bin/activate`), đang đứng ở thư mục `openpi/`.

```bash
VENV_SITE=/home/hung/hungvd27/Pi05_ONNX_Tensorrt/.venv/lib/python3.10/site-packages

# 1. Torch CUDA thật cho Jetson (KHÔNG dùng `pip install torch` thường — ra bản CPU-only)
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch==2.8.0 torchvision==0.23.0

# 2. Cho venv thấy TensorRT python bindings đã cài qua apt (python3-libnvinfer)
echo "/usr/lib/python3.10/dist-packages" > "$VENV_SITE/_system_dist_packages.pth"

# 3. Google Cloud Storage download helper cho openpi.shared.download
pip install fsspec gcsfs tqdm-loggable

# 4. Core deps của openpi (đúng version pin quan trọng)
pip install safetensors tyro einops beartype jaxtyping augmax dm-tree equinox \
  flatbuffers gym-aloha imageio ml-collections numpydantic opencv-python pillow \
  sentencepiece wandb chex transformers polars "orbax-checkpoint==0.11.13" \
  "numpy>=1.26,<2.0.0"   # xem lưu ý về numpy ở cuối mục này

# 5. HuggingFace datasets (dùng bởi lerobot) + toàn bộ dependency của lerobot
pip install "datasets==3.6.0" av deepdiff diffusers draccus flask gdown \
  "gymnasium==0.29.1" h5py jsonlines numba omegaconf opencv-python-headless \
  pymunk pynput pyzmq rerun-sdk termcolor "zarr>=2.17.0,<3.0.0"
  # Lưu ý: KHÔNG dùng zarr 3.x — zarr==3.0.8 (bản pin gốc trong uv.lock) yêu cầu
  # Python >=3.11, không cài được trên venv 3.10.

# 6. lerobot đúng commit pin trong uv.lock (không phải bản mới nhất trên PyPI/GitHub)
pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"

# 7. openpi-client + openpi (project chính) ở chế độ editable
pip install -e packages/openpi-client
pip install --ignore-requires-python -e .   # venv là 3.10, pyproject.toml đòi >=3.11

# 8. Patch transformers với các file custom của openpi (Gemma/SigLIP/PaliGemma)
cp -r ./src/openpi/models_pytorch/transformers_replace/* "$VENV_SITE/transformers/"

# 9. Chỉ cần cho export FP8 (bỏ qua nếu chỉ dùng FP16)
pip install "nvidia-modelopt==0.33.1" onnx onnxslim lief

# 10. Chỉ cần cho bước Verify (so sánh PyTorch/ONNX/TensorRT) ở dưới
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu==1.24.0
```

Fix riêng cho file `src/openpi/training/data_loader.py` (nếu bạn có sửa dở import
`lerobot.datasets.lerobot_dataset` sang cấu trúc lerobot bản mới — xem mục
"Các fix môi trường khác" bên dưới):

```bash
mkdir -p "$VENV_SITE/lerobot/datasets"
touch "$VENV_SITE/lerobot/datasets/__init__.py"
cat > "$VENV_SITE/lerobot/datasets/lerobot_dataset.py" <<'EOF'
from lerobot.common.datasets.lerobot_dataset import *  # noqa: F401,F403
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: F401
EOF
```

> **Lưu ý về `numpy`:** đây là điểm dễ vỡ nhất của cả setup. `torch` (build cho
> Jetson) được compile với numpy 1.x nên **cảnh báo** (không lỗi) khi chạy với
> numpy 2.x; nhưng `onnx`/`ml_dtypes`/`onnxruntime-gpu==1.23.0` lại **bắt buộc**
> numpy 2.x (hard fail nếu numpy <2). Bước cài `nvidia-modelopt`/`onnx` ở trên sẽ
> tự kéo numpy lên lại 2.x — cứ để vậy, đừng cố ép về `<2.0` cho toàn bộ venv.
> Hệ quả phụ: `torch.Tensor.numpy()` bị hỏng ở numpy 2.x (xem mục Verify) —
> dùng `tensor.tolist()` rồi `np.array(...)` thay cho `.numpy()` khi cần chuyển
> tensor GPU/CPU sang numpy trong venv này.

## Quy trình convert

### 1. Tải checkpoint JAX

```bash
export CONFIG_NAME=pi05_libero
python -c "
from openpi.shared import download
print(download.maybe_download(f'gs://openpi-assets/checkpoints/{'$CONFIG_NAME'}'))
"
```

Nếu tải chậm bất thường do máy không phải instance GCP, `google-auth` sẽ cố ping
metadata server và timeout nhiều lần. Đặt `NO_GCE_CHECK=true` (chữ thường, bắt
buộc — xem `google/auth/compute_engine/_metadata.py`) để bỏ qua bước dò này.

### 2. JAX → PyTorch

```bash
# Patch transformers package (chỉ cần 1 lần cho mỗi lần cài lại transformers)
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
  /home/hung/hungvd27/Pi05_ONNX_Tensorrt/.venv/lib/python3.10/site-packages/transformers/

python examples/convert_jax_model_to_pytorch.py \
  --config-name pi05_libero \
  --checkpoint-dir /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --output-path /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

Output: `model.safetensors` (~6.8GB, 3.62B params) + `config.json` + `assets/`.

### 3. PyTorch → ONNX (FP16)

```bash
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --output_path /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --config_name pi05_libero \
  --precision fp16
```

Output: `onnx/model_fp16.onnx` (+ `.onnx.data`, ~6.1GB tổng).

### 3b. PyTorch → ONNX (FP8, có calibration)

```bash
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --output_path /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --config_name pi05_libero \
  --precision fp8 \
  --quantize_attention_matmul
```

Script tự tải dataset `physical-intelligence/libero` (LeRobot) để calibrate; nếu
load dataset lỗi nó **tự fallback sang dummy-input calibration** (xem Bug 5) —
không phải lỗi nghiêm trọng, chỉ giảm độ chính xác quantize.

### 4. ONNX → TensorRT

```bash
ulimit -c 0   # tắt core dump, tránh apport treo máy nếu trtexec crash (xem Bug 1)
MAX_BATCH=1 OPT_BATCH=1 MIN_BATCH=1 ./deployment_scripts/build_engine.sh \
  <path/to/model.onnx> \
  <path/to/output.engine>
```

Kết quả trên Orin: **FP16 build PASSED** (~6.1GB engine, ~300ms/inference cho
10 bước denoising). **FP8 build KHÔNG THỂ chạy trên Orin** — xem Bug 6.

### 5. Verify: so sánh output PyTorch vs ONNXRuntime vs TensorRT

`deployment_scripts/compare_backends.py` chạy `sample_actions` (10 bước
denoising) qua cả 3 backend với **cùng một bộ input cố định** (seed cố định),
rồi so MSE/MAE/MaxAbsDiff/CosineSim giữa từng cặp — để xác nhận convert không
làm sai lệch output ngoài phạm vi làm tròn số học bình thường.

Yêu cầu thêm `onnxruntime-gpu` (xem mục cài đặt môi trường ở trên):

```bash
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu==1.24.0

python deployment_scripts/compare_backends.py \
  --checkpoint-dir /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --onnx-path /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/onnx_fixed/onnx/model_fp16.onnx \
  --engine-path /home/hung/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/model_fp16_seq200.engine \
  --config-name pi05_libero
```

Kết quả đo được trên Orin (input `(1, 9, 224, 224)` ảnh + 200 token ngôn ngữ,
output actions `(1, 10, 32)` = 320 giá trị):

| Backend | Thời gian/inference |
|---|---|
| PyTorch (CUDA, fp16, eager) | 1474.6 ms |
| ONNXRuntime (CUDAExecutionProvider) | 929.3 ms |
| **TensorRT engine** | **385.3 ms** (~3.8× nhanh hơn PyTorch, ~2.4× nhanh hơn ORT) |

| So sánh | MSE | MAE | MaxAbsDiff | CosineSim |
|---|---|---|---|---|
| PyTorch vs ONNXRuntime | 4.03e-08 | 1.46e-04 | 9.77e-04 | 0.999999 |
| PyTorch vs TensorRT | 7.88e-08 | 1.86e-04 | 1.46e-03 | 0.999999 |
| ONNXRuntime vs TensorRT | 4.30e-08 | 1.50e-04 | 9.77e-04 | 0.999999 |

Sai số ở mức 1e-7 (MSE) / cosine similarity ~1.0 — hoàn toàn nằm trong phạm vi
làm tròn FP16 bình thường, xác nhận cả 3 dạng model cho ra kết quả nhất quán
với nhau.

> Script dùng `patch_model_for_export()` trong `pytorch_to_onnx.py` để chạy
> PyTorch qua đúng các hook (`sample_actions_hook`, `embed_prefix`/`embed_suffix`
> "ONNX-safe") đã dùng lúc export — so trực tiếp `PI0Pytorch.sample_actions` gốc
> (chưa patch) với ONNX/TensorRT sẽ không phải so sánh đúng nghĩa (hai code path
> khác nhau).

---

## Danh sách bug đã gặp và cách fix

### Bug 1 — `trtexec` crash (SIGSEGV) khi build engine với graph đã fuse

**Triệu chứng:** `trtexec` crash giữa chừng, kernel gọi `apport` để dump core
(process ~18GB) khiến máy gần như treo hàng chục phút.

**Nguyên nhân:** graph ONNX xuất ra với `perf_opts=True` (mặc định của
`pytorch_to_onnx.py`, gồm `chunked_ae_attention`, `gqa_zero_copy`,
`fold_time_constants`, `fold_adaln_dense`, `vit_view_batch`,
`fuse_ae_projections`) tạo ra một "Myelin ForeignNode" phức tạp mà TensorRT
10.3.0 xử lý không ổn định.

**Fix:**
- Luôn chạy `trtexec` với `ulimit -c 0` trước để tắt core dump.
- Export ONNX với `--no-perf_opts` để lấy graph đơn giản hơn (không dùng fusion
  tối ưu, chấp nhận chậm hơn một chút khi infer) — hoặc dùng ONNX đã fuse nếu về
  sau TensorRT bản mới hơn xử lý được.

### Bug 2 — `IElementWiseLayer /And_1: broadcast dimensions must be conformable`

**Triệu chứng:** `trtexec` fail cả ở bước parse lẫn build với lỗi shape ở layer
`And` (dùng để build attention mask 2D), xảy ra bất kể có fuse hay không, bất kể
`--stronglyTyped` hay không, bất kể batch size tĩnh hay động.

**Nguyên nhân gốc (2 lớp):**

1. **`src/openpi/models_pytorch/pi0_pytorch.py`** — `embed_prefix`/`embed_suffix`
   dùng `bsize, num_img_embs = img_emb.shape[:2]` rồi dùng trực tiếp trong
   `.expand(bsize, num_img_embs)`. Khi trace ONNX, các giá trị này có thể bị giữ
   dạng "dynamic" (Shape-derived) thay vì hằng số, trong khi cùng giá trị đó lại
   được dùng qua `[0] * num_img_embs` (Python list) để build `att_masks` — buộc
   phải resolve về số nguyên cụ thể ngay lập tức. Hai nhánh cùng giá trị số học
   nhưng khác "identity" trong đồ thị ONNX.
2. **`deployment_scripts/build_engine.sh`** — hardcode
   `MAX_SEQ_LEN=208` (làm tròn lên bội số 16 cho hiệu năng), nhưng
   `pytorch_to_onnx.py` dùng đúng `model_config.max_token_len` (= **200** cho
   `pi05_libero`) khi tạo dummy input lúc export. Model ONNX được bake cứng với
   tổng sequence length tính từ 200 lang tokens, còn profile shape truyền cho
   `trtexec` lại giả định 208 tokens → lệch 8 token → mismatch shape thật giữa
   nhánh mask "hằng số" (968) và nhánh mask tính từ input thật theo profile
   (976).

**Fix:**

`src/openpi/models_pytorch/pi0_pytorch.py` — ép kiểu `int()` ngay khi lấy kích
thước từ tensor, để ONNX export bake thành hằng số nhất quán (không đổi hành vi
runtime, chỉ ảnh hưởng lúc trace):

```python
# embed_prefix
bsize, num_img_embs = (int(d) for d in img_emb.shape[:2])
...
num_lang_embs = int(lang_emb.shape[1])
...
bsize = int(pad_masks.shape[0])

# embed_suffix
bsize = int(state_emb.shape[0])
...
bsize, action_time_dim = (int(d) for d in action_time_emb.shape[:2])
```

`deployment_scripts/build_engine.sh` — sửa `MAX_SEQ_LEN` cho khớp với
`max_token_len` thật của config đang export (200 cho `pi05_libero`), có thể
override qua biến môi trường:

```bash
# Must match the model config's max_token_len (pytorch_to_onnx.py bakes this length
# into the ONNX graph's attention-mask constants during export). A mismatch here
# makes TensorRT reject the graph with "IElementWiseLayer /And_*: broadcast
# dimensions must be conformable".
MAX_SEQ_LEN="${MAX_SEQ_LEN:-200}"
```

> Cách fix ở `pi0_pytorch.py` (ép `int()`) tự nó **không đủ** để hết lỗi — nó
> giúp qua được bước *parse*, nhưng lỗi vẫn xảy ra ở bước *build* cho tới khi
> sửa luôn `MAX_SEQ_LEN` trong `build_engine.sh`. Cần cả hai.

### Bug 3 — `orbax-checkpoint` API thay đổi giữa các version

**Triệu chứng:** `TypeError: 'StepMetadata' object is not subscriptable` tại
`src/openpi/models/model.py:316` (`metadata["params"]`).

**Nguyên nhân:** bản `orbax-checkpoint` cài tự do (mới nhất, `0.11.39`) đã đổi
`PyTreeCheckpointer.metadata()` trả về object `StepMetadata` thay vì dict thô.
`uv.lock` của project pin đúng `orbax-checkpoint==0.11.13`.

**Fix:** `pip install "orbax-checkpoint==0.11.13"`.

### Bug 4 — `nvidia-modelopt` đổi cấu trúc `FP8_DEFAULT_CFG`

**Triệu chứng:** export FP8 fail với
`TypeError: list indices must be integers or slices, not str` tại
`deployment_scripts/pytorch_to_onnx.py:764`
(`quant_cfg["quant_cfg"]["nn.Conv2d"] = ...`).

**Nguyên nhân:** `nvidia-modelopt==0.46.1` (bản mới nhất trên PyPI) đổi
`mtq.FP8_DEFAULT_CFG["quant_cfg"]` từ `dict` sang `list`. Script được viết cho
bản cũ hơn — đúng version pin trong `deployment_scripts/pyproject.toml` (extras
`[deploy]`) là **`nvidia-modelopt==0.33.1`**.

**Fix:** `pip install "nvidia-modelopt==0.33.1"`.

### Bug 5 — Dataset calibration `physical-intelligence/libero` lỗi format

**Triệu chứng:** load dataset LeRobot để calibrate FP8 fail với
`TypeError: list indices must be integers or slices, not str` (khác bug 4, xảy
ra trong `lerobot`, không phải `modelopt`), kèm warning:

```
The dataset you requested (physical-intelligence/libero) is in 2.0 format...
Revision v2.1 for physical-intelligence/libero not found, using version v2.0
```

**Nguyên nhân:** dataset trên HuggingFace Hub ở format v2.0 (global stats),
version `lerobot` đang cài đọc theo format mới hơn (per-episode stats) không
tương thích hoàn toàn.

**Fix:** không cần sửa gì — `deployment_scripts/calibration_data.py` +
`pytorch_to_onnx.py` đã có sẵn cơ chế fallback: bắt exception, in
`"Falling back to dummy inputs for calibration"` và tiếp tục export bằng dummy
input. Model FP8 vẫn export được, chỉ là quantize dựa trên dữ liệu giả thay vì
dữ liệu thật (có thể ảnh hưởng độ chính xác, không ảnh hưởng khả năng chạy).

> Lưu ý: bước tải dataset này tải ~50-60GB trước khi fail, từng làm đầy ổ đĩa.
> Nếu disk hạn chế, cân nhắc xoá `~/.cache/huggingface/lerobot` và
> `~/.cache/huggingface/hub/datasets--physical-intelligence--libero` sau khi
> export FP8 xong (dataset không dùng lại được do bug format ở trên).

### Bug 6 — FP8 TensorRT engine không build được trên Orin (giới hạn phần cứng)

**Triệu chứng:**

```
Error Code 9: API Usage Error (Networks with FP8 Q/DQ layers require hardware with FP8 support.)
```

**Nguyên nhân:** GPU Jetson AGX Orin dùng kiến trúc **Ampere**, không có FP8
Tensor Core. FP8 chỉ được hỗ trợ từ Hopper/Ada trở lên (hoặc Blackwell trên
Jetson Thor — đúng target gốc của `deployment_scripts/`). Đây là giới hạn
silicon, không có cách fix bằng software.

**Kết luận:** trên Jetson AGX Orin, **FP16 là lựa chọn tối ưu khả dụng cao nhất**
cho pipeline này. Muốn dùng FP8/NVFP4 thật sự cần chạy trên Jetson Thor.

### Bug 7 — `torch.Tensor.numpy()` báo `RuntimeError: Numpy is not available`

**Triệu chứng:** gọi `.numpy()` trên tensor (kể cả tensor CPU) fail cứng, dù lúc
import `torch` chỉ thấy **warning** (không phải lỗi) về numpy ABI:

```
UserWarning: Failed to initialize NumPy: A module that was compiled using
NumPy 1.x cannot be run in NumPy 2.2.6 as it may crash...
```

**Nguyên nhân:** bản `torch==2.8.0` cài từ index Jetson được build với numpy
1.x C-API, nhưng venv cần numpy 2.x cho `onnx`/`ml_dtypes`/`onnxruntime-gpu`
(xem lưu ý ở mục cài đặt). Lúc import thì torch chỉ cảnh báo, nhưng khi thực sự
gọi cầu nối C++ `Tensor.numpy()` thì fail.

**Fix:** không dùng `.numpy()` trực tiếp trên tensor trong venv này — chuyển
qua `tensor.detach().to("cpu", torch.float32).tolist()` rồi mới
`np.array(...)`. Xem hàm `to_numpy()` trong
`deployment_scripts/compare_backends.py`.

### Các fix môi trường khác (không phải bug logic, chỉ là thiếu setup)

- Thiếu hàng loạt package Python: `fsspec`, `tqdm-loggable`, `gcsfs`,
  `safetensors`, `tyro`, `einops`, `beartype`, `jaxtyping`, `augmax`, `dm-tree`,
  `equinox`, `flatbuffers`, `gym-aloha`, `imageio`, `ml-collections`,
  `numpydantic`, `opencv-python`, `pillow`, `sentencepiece`, `wandb`, `chex`,
  `transformers`, `polars`, `datasets`, và toàn bộ dependency của `lerobot`
  (`av`, `deepdiff`, `diffusers`, `draccus`, `flask`, `gdown`, `gymnasium`,
  `h5py`, `jsonlines`, `numba`, `omegaconf`, `opencv-python-headless`,
  `pymunk`, `pynput`, `pyzmq`, `rerun-sdk`, `termcolor`, `torchvision`, `zarr`).
  → cài đúng version pin trong `uv.lock` khi có thể; `zarr` phải dùng nhánh
  `2.x` (`zarr>=2.17.0,<3.0.0`) vì `zarr==3.0.8` (bản pin gốc) yêu cầu Python
  ≥3.11 trong khi venv là 3.10.
- Gói `openpi` (project chính, `src/openpi`) và `openpi-client` chưa được cài ở
  chế độ editable → `pip install -e .` (thêm `--ignore-requires-python` vì
  venv Python 3.10 < yêu cầu `>=3.11` của `pyproject.toml`) và
  `pip install -e packages/openpi-client`.
- `lerobot` cần cài đúng commit pin trong `uv.lock`:
  `pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"`.
- File `src/openpi/training/data_loader.py` (đang sửa dở, chưa commit) import
  `lerobot.datasets.lerobot_dataset` (cấu trúc `lerobot` bản rất mới, đã chuyển
  sang `src/` layout) trong khi bản `lerobot` cài theo pin lại có cấu trúc cũ
  `lerobot.common.datasets.lerobot_dataset`. Đã tạo shim module
  `lerobot/datasets/lerobot_dataset.py` trong site-packages để re-export từ
  `lerobot.common.datasets.lerobot_dataset`, tránh phải sửa code đang dở của
  bạn hay đổi version `lerobot`.
- TensorRT python bindings (`tensorrt`) cài qua apt nằm ở
  `/usr/lib/python3.10/dist-packages`, venv mặc định không thấy — đã thêm file
  `.pth` trỏ tới thư mục đó trong `site-packages` của venv.

## Tóm tắt file đã sửa / thêm mới

| File | Thay đổi |
|---|---|
| `src/openpi/models_pytorch/pi0_pytorch.py` | Ép `int()` cho shape dùng trong mask construction (Bug 2) |
| `deployment_scripts/build_engine.sh` | `MAX_SEQ_LEN` 208 → 200, cho phép override qua env var (Bug 2) |
| `deployment_scripts/compare_backends.py` | **Mới** — script verify so sánh PyTorch/ONNXRuntime/TensorRT (mục Verify) |
| `deployment_scripts/CONVERSION_GUIDE.md` | **Mới** — tài liệu này |

Các version package cần đúng pin (không phải "mới nhất"):

| Package | Version đúng | Vì sao |
|---|---|---|
| `orbax-checkpoint` | `0.11.13` | Bug 3 |
| `nvidia-modelopt` | `0.33.1` | Bug 4 |
| `zarr` | `>=2.17.0,<3.0.0` | `3.0.8` cần Python ≥3.11 |
| `torch` / `torchvision` | `2.8.0` / `0.23.0` từ `pypi.jetson-ai-lab.io/jp6/cu126` | Cần bản CUDA thật cho Orin, không phải CPU-only từ PyPI |
| `onnxruntime-gpu` | `1.24.0` từ `pypi.jetson-ai-lab.io/jp6/cu126` | Bug 7 — bản `1.23.0` build với numpy 1.x, hard-fail trên numpy 2.x |
