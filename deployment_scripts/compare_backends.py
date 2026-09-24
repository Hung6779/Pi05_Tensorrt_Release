"""Compare PyTorch / ONNXRuntime / TensorRT outputs for a converted pi0 checkpoint.

Runs `sample_actions` through all three backends with the same fixed random inputs
(images/tokens/state/noise) and reports MSE/MAE/MaxAbsDiff/CosineSim between every
pair, plus per-backend inference time. See CONVERSION_GUIDE.md "Verify" section.

Usage:
    python deployment_scripts/compare_backends.py \
        --checkpoint-dir /path/to/pi05_libero_pytorch \
        --onnx-path /path/to/model_fp16.onnx \
        --engine-path /path/to/model_fp16.engine \
        --config-name pi05_libero
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SEED = 1234
NUM_IMAGES = 3  # len(openpi.models.model.IMAGE_KEYS) - fixed across all pi0 configs
IMAGE_SIZE = 224
NUM_STEPS = 10

PALIGEMMA_VOCAB_SIZE = 257152  # from openpi.models.gemma, used only for bounding random tokens


def to_numpy(t: torch.Tensor) -> np.ndarray:
    # torch's native tensor->numpy bridge (Tensor.numpy()) is broken under this venv's
    # numpy 2.x (torch was built against numpy 1.x headers - see the "Failed to
    # initialize NumPy" warning on import). Round-tripping through a Python list avoids
    # that C-API bridge entirely.
    return np.array(t.detach().to("cpu", torch.float32).tolist(), dtype=np.float32)


def get_model_dims(config_name):
    """Read action_dim/action_horizon/max_token_len straight from the config instead of
    hardcoding them - different checkpoints (e.g. pi05_ur10e_cup's action_horizon=50 vs
    pi05_libero's 10) will otherwise silently build the wrong-shaped dummy inputs."""
    import openpi.training.config as _config

    model_cfg = _config.get_config(config_name).model
    return model_cfg.action_dim, model_cfg.action_horizon, model_cfg.max_token_len


def make_inputs(action_dim, action_horizon, max_token_len):
    g = torch.Generator(device="cpu").manual_seed(SEED)
    images = torch.randn(1, NUM_IMAGES * 3, IMAGE_SIZE, IMAGE_SIZE, generator=g, dtype=torch.float32).to(
        torch.float16
    )
    img_masks = torch.ones(1, NUM_IMAGES, dtype=torch.bool)
    lang_tokens = torch.randint(0, 32000, (1, max_token_len), generator=g, dtype=torch.long)
    lang_masks = torch.ones(1, max_token_len, dtype=torch.bool)
    state = torch.randn(1, action_dim, generator=g, dtype=torch.float32).to(torch.float16)
    noise = torch.randn(1, action_horizon, action_dim, generator=g, dtype=torch.float32).to(torch.float16)
    return images, img_masks, lang_tokens, lang_masks, state, noise


def run_pytorch(inputs, checkpoint_dir, config_name):
    import dataclasses

    import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
    import openpi.training.config as _config
    import safetensors.torch
    from openpi.models.model import IMAGE_KEYS, Observation

    from deployment_scripts.pytorch_to_onnx import ExportOptions, patch_model_for_export

    config = _config.get_config(config_name)
    # Disable torch.compile for this check: it changes nothing numerically (same eager
    # graph, just JIT-compiled) but adds a multi-minute autotune pass on first call.
    model_cfg = dataclasses.replace(config.model, pytorch_compile_mode=None)
    model = pi0_pytorch.PI0Pytorch(config=model_cfg)
    safetensors.torch.load_model(model, f"{checkpoint_dir}/model.safetensors", strict=False)
    model = model.to("cuda").to(torch.float16).eval()

    # onnx_fixed/onnx/model_fp16.onnx was exported with --no-perf_opts, so match that
    # here: patch_model_for_export swaps in the same ONNX-safe sample_actions/embed_*
    # hooks (and sets model.compute_dtype) that the export actually traced, so this is
    # a true apples-to-apples reference rather than the un-patched eager implementation.
    opts = ExportOptions(perf_opts=False, quantize_attention_matmul=False)
    model = patch_model_for_export(model, opts, compute_dtype=torch.float16)

    images, img_masks, lang_tokens, lang_masks, state, noise = [x.to("cuda") for x in inputs]

    observation = Observation(
        images={IMAGE_KEYS[i]: images[:, i * 3 : (i + 1) * 3] for i in range(len(IMAGE_KEYS))},
        image_masks={IMAGE_KEYS[i]: img_masks[:, i] for i in range(len(IMAGE_KEYS))},
        state=state,
        tokenized_prompt=lang_tokens,
        tokenized_prompt_mask=lang_masks,
    )
    with torch.no_grad():
        t0 = time.time()
        actions = model.sample_actions(images.device, observation, noise=noise, num_steps=NUM_STEPS)
        torch.cuda.synchronize()
        dt = time.time() - t0
    return to_numpy(actions), dt


def run_onnxruntime(inputs, onnx_path):
    import onnxruntime as ort

    images, img_masks, lang_tokens, lang_masks, state, noise = inputs
    feed = {
        "images": np.array(images.tolist(), dtype=np.float16),
        "img_masks": np.array(img_masks.tolist(), dtype=bool),
        "lang_tokens": np.array(lang_tokens.tolist(), dtype=np.int64),
        "lang_masks": np.array(lang_masks.tolist(), dtype=bool),
        "state": np.array(state.tolist(), dtype=np.float16),
        "noise": np.array(noise.tolist(), dtype=np.float16),
    }
    sess_opts = ort.SessionOptions()
    providers = [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    print("  ORT providers in use:", sess.get_providers())
    input_names = {i.name for i in sess.get_inputs()}
    feed = {k: v for k, v in feed.items() if k in input_names}
    t0 = time.time()
    (out,) = sess.run(["actions"], feed)
    dt = time.time() - t0
    return out.astype(np.float32), dt


def run_tensorrt(inputs, engine_path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    images, img_masks, lang_tokens, lang_masks, state, noise = [x.to("cuda").contiguous() for x in inputs]
    name_to_tensor = {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "state": state,
        "noise": noise,
    }

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if name in name_to_tensor:
            context.set_input_shape(name, tuple(name_to_tensor[name].shape))

    output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            output_name = name
            out_shape = context.get_tensor_shape(name)
            out_dtype_trt = engine.get_tensor_dtype(name)
    torch_dtype = torch.float16 if out_dtype_trt == trt.DataType.HALF else torch.float32
    output = torch.empty(tuple(out_shape), dtype=torch_dtype, device="cuda")

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if name in name_to_tensor:
            context.set_tensor_address(name, name_to_tensor[name].data_ptr())
    context.set_tensor_address(output_name, output.data_ptr())

    stream = torch.cuda.Stream()
    t0 = time.time()
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    dt = time.time() - t0
    return to_numpy(output), dt


def compare(name_a, a, name_b, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    diff = a - b
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    max_abs = np.max(np.abs(diff))
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = float(np.dot(a, b) / denom) if denom > 0 else float("nan")
    print(f"  {name_a} vs {name_b}: MSE={mse:.6e}  MAE={mae:.6e}  MaxAbsDiff={max_abs:.6e}  CosineSim={cos:.6f}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True, help="PyTorch checkpoint dir (has model.safetensors)")
    p.add_argument("--onnx-path", required=True, help="Path to the exported .onnx file")
    p.add_argument("--engine-path", required=True, help="Path to the built TensorRT .engine file")
    p.add_argument("--config-name", default="pi05_libero", help="openpi training config name")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed for the fixed dummy inputs")
    p.add_argument(
        "--backend",
        choices=["pytorch", "onnx", "tensorrt"],
        default=None,
        help=argparse.SUPPRESS,  # internal: used for the isolated subprocess re-invocation below
    )
    p.add_argument("--work-dir", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def _run_single_backend(args, inputs):
    """Run exactly one backend and dump its (action, elapsed_seconds) to <work_dir>/<backend>.npz.
    Invoked as a subprocess so each backend's CUDA context (PyTorch's caching allocator,
    ONNXRuntime's own CUDA memory arena, or a TensorRT engine) is fully torn down by the
    OS on process exit - see the module docstring / free_gpu_memory() for why an
    in-process gc.collect()+empty_cache() isn't enough to avoid OOM between backends.
    """
    action_dim, action_horizon, max_token_len = get_model_dims(args.config_name)
    inputs = make_inputs(action_dim, action_horizon, max_token_len)
    if args.backend == "pytorch":
        out, dt = run_pytorch(inputs, args.checkpoint_dir, args.config_name)
    elif args.backend == "onnx":
        out, dt = run_onnxruntime(inputs, args.onnx_path)
    else:
        out, dt = run_tensorrt(inputs, args.engine_path)
    np.savez(os.path.join(args.work_dir, f"{args.backend}.npz"), out=out, dt=dt)


def main():
    args = parse_args()
    global SEED
    SEED = args.seed

    if args.backend is not None:
        # Isolated subprocess mode: inputs are rebuilt from the same --seed, so they're
        # identical across the three invocations without needing to serialize them.
        _run_single_backend(args, None)
        return

    action_dim, action_horizon, max_token_len = get_model_dims(args.config_name)
    print(f"Config '{args.config_name}': action_dim={action_dim} action_horizon={action_horizon} max_token_len={max_token_len}")

    import subprocess
    import tempfile

    results = {}
    labels = {"pytorch": "PyTorch (CUDA, fp16)", "onnx": "ONNXRuntime (CUDAExecutionProvider)", "tensorrt": "TensorRT engine"}
    with tempfile.TemporaryDirectory() as work_dir:
        for backend in ["pytorch", "onnx", "tensorrt"]:
            print(f"\nRunning {labels[backend]} (isolated subprocess)...")
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--checkpoint-dir", args.checkpoint_dir,
                "--onnx-path", args.onnx_path,
                "--engine-path", args.engine_path,
                "--config-name", args.config_name,
                "--seed", str(args.seed),
                "--backend", backend,
                "--work-dir", work_dir,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stdout[-4000:])
                print(proc.stderr[-4000:])
                raise RuntimeError(f"{backend} subprocess failed with exit code {proc.returncode}")
            data = np.load(os.path.join(work_dir, f"{backend}.npz"))
            out, dt = data["out"], float(data["dt"])
            results[backend] = out
            print(f"  actions shape={out.shape} dtype={out.dtype} time={dt*1000:.1f} ms")
            print("  sample[0,0,:5] =", out[0, 0, :5])

    print("\n=== Comparison (fp32 accumulation) ===")
    compare("PyTorch", results["pytorch"], "ONNXRuntime", results["onnx"])
    compare("PyTorch", results["pytorch"], "TensorRT", results["tensorrt"])
    compare("ONNXRuntime", results["onnx"], "TensorRT", results["tensorrt"])


if __name__ == "__main__":
    main()
