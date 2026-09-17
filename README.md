# SAM-3 deployment scripts

Two independent pieces:

1. **`onnx_export_and_trt/`** — exports `facebook/sam3` (HuggingFace `transformers`) to ONNX, then
   builds the TensorRT FP16 engines actually used at inference time.
2. **`ros_node/`** — the ROS node and supporting Python that run those engines against OAK camera
   footage for PPE (helmet) compliance detection.

No data (images, rosbags, cached tensors, model weights) is included — only code. Everything here
reads paths from local flags/constants (`--bags-dir`, `--output-dir`, `onnx_weights/`, etc.) that
you point at your own environment. `facebook/sam3` is a public HuggingFace model; nothing here
embeds or requires an API key.

## Environment

What this actually ran on, checked directly against the working install rather than guessed:

- **OS**: Linux, with **ROS Noetic** installed at the system level (`/opt/ros/noetic`) — ROS Noetic
  targets Ubuntu 20.04's system Python 3 (3.8), which is too old for current `transformers`/SAM-3.
- **GPU**: NVIDIA, CUDA 12.1 toolkit, driver 535+ (tested on an RTX 4090, 24GB VRAM). TensorRT engine
  *building* (not inference) is the peak memory moment — `WORKSPACE_GB = 4` in the build scripts: raise
  it if a build fails for lack of workspace, lower it if you're on a smaller card.
- **Python**: a **separate 3.10 virtualenv**, isolated from ROS's own system Python (this project
  used `pyenv` but any 3.9+ env manager works the same way). This is the crux of
  the setup: you need a modern Python for `torch`/`transformers`/`tensorrt`, but ROS Noetic's own
  `rospy`/`rosbag` are only installed against the system Python.
- **Bridging ROS into that virtualenv**: `source /opt/ros/noetic/setup.bash` first (sets `PYTHONPATH`
  to ROS's system `dist-packages`, where `rospy`/`rosbag`/`sensor_msgs`/`std_msgs` live), then run
  scripts with the **virtualenv's own `python3` binary**, not `python`/`rosrun`. The virtualenv then
  sees both its own pip-installed packages *and* ROS's system packages on `sys.path`. Three extra
  pip packages are needed for `rospy`/`rosbag` to actually import cleanly from a non-system Python:
  `rospkg`, `pycryptodomex`, `python-gnupg` (all in `requirements.txt`).
- **Package versions**: `requirements.txt` covers everything (export + inference);
  `deployment_req.txt` is the smaller inference-only subset (skip it if you're only ever running
  `ros_node/` against engines someone else already built). Two load-bearing details either way —
  - `transformers` needs to come from a **git commit**, not PyPI: SAM-3 support isn't in a stable
    release yet. This ran against
    `git+https://github.com/huggingface/transformers.git@453a246113254b173e044f13a22c9e6b56259ae`
    (current `main` should work too, but wasn't what was actually tested).
  - `tensorrt==10.7.0` (the `tensorrt-cu12-*` wheels) matched against CUDA 12.1 — mixing a
    TensorRT build against a different CUDA major version is a common source of engine-build or
    load failures.
- **Run pattern**, every script in both folders:
  ```bash
  source /opt/ros/noetic/setup.bash
  /path/to/your/venv/bin/python3 <script>.py [args]
  ```
  (the export scripts under `onnx_export_and_trt/` don't actually need ROS on `PYTHONPATH`, but
  sourcing it first is harmless and keeps one run pattern for everything).

## `onnx_export_and_trt/` — run in this exact order

Each script writes into `./onnx_weights/` (created if missing) relative to wherever you run it.

| # | Script | Produces | Notes |
|---|--------|----------|-------|
| 1 | `export_sam3_cached.py` | `sam3_encoder.onnx`, `sam3_decoder.onnx` | Splits SAM-3 into a vision encoder (run once per image) and a decoder (run once per text prompt) so multi-prompt inference doesn't re-run the encoder every time (~2.2x speedup). Traces both with `torch.onnx.export` (opset 17) using a public COCO sample image. Runs on CPU for export-compatibility reasons — no GPU needed for this step. |
| 2 | `build_trt_split.py` | `sam3_encoder_fp16.engine`, `sam3_decoder_fp16.engine` | Builds FP16 TensorRT engines from the two ONNX files above. The encoder engine is the one actually used downstream; the single-prompt decoder engine here is superseded by the batched one below but is built as a side effect of the same script. Requires `tensorrt` (`pip install tensorrt` or an NGC container) and a GPU. |
| 3 | `export_onnx_sam3_batched.py` | `sam3_decoder_batched.onnx` | Re-exports the decoder to accept a *batch* of prompts in one call (`[N, ...]` inputs/outputs, dynamic `num_prompts` and `sequence_length` axes) instead of one prompt at a time — this is what lets the pipeline query e.g. `"person"`, `"hard hat"`, `"head"` in a single decoder call. Also CPU-only, also needs the encoder output shapes as trace inputs (regenerated internally, not read from step 1's output). |
| 4 | `export_trt_sam3_batched.py` | `sam3_decoder_batched_fp16.engine` | Builds the FP16 TensorRT engine for the batched decoder ONNX. Optimization profile covers 1-8 prompts (optimized for 5); if you need more than 8 simultaneous prompts, raise `max_s` in the profile before building. |

**What's actually loaded at inference time** (see `ros_node/mask_extraction_node.py`,
`Sam3TrtBatchedPipeline`): `sam3_encoder_fp16.engine` (from step 2) + `sam3_decoder_batched_fp16.engine`
(from step 4). The non-batched `sam3_decoder_fp16.engine` from step 2 isn't used by the current
pipeline but costs nothing extra to build alongside the encoder.

Dependencies: `torch`, `transformers` (with SAM-3 support), `tensorrt`, `pillow`, `requests` — see
`requirements.txt` and the Environment section above for exact versions and the CPU/GPU split.

## `ros_node/` — the inference pipeline and its outputs

| File | Role |
|------|------|
| `mask_extraction_node.py` | Live ROS node: subscribes to an OAK camera image topic, runs the TensorRT SAM-3 pipeline (`Sam3TrtBatchedPipeline` — one encoder call + one batched decoder call per frame), applies the helmet-worn rule, publishes an annotated image + JSON results, and reports achieved FPS against the incoming topic's rate. Also defines the shared pipeline class, prompts, and verdict logic that `batch_process_bags.py` reuses. |
| `body_segmentation.py` | Per-person geometry: matches each SAM-3 "head and upper body" detection (used directly as the person anchor — no separate "person" prompt) to its "head" and "upper garment" detections, with a small evidence-based reflection-area floor as the only real filter. Also has a standalone `main()` demo against a local image folder (not part of the live pipeline). |
| `batch_process_bags.py` | Offline batch tool: for each input `.bag`, writes a new bag with all original topics preserved plus annotated image / binary mask / JSON-results topics, and compiles a per-bag MP4. Reuses the exact same pipeline and rules as the live node. |
| `visualize_rgb_and_mask.py` | Standalone QA tool: renders a side-by-side (RGB \| binary mask) MP4 from an already-produced `batch_process_bags.py` output bag, for visually checking mask quality. |

Dependencies: `rospy`, `rosbag`, `sensor_msgs`, `std_msgs` (from ROS Noetic, via `PYTHONPATH` — see
Environment above), plus **`deployment_req.txt`** — a smaller set than `requirements.txt`, for
running against already-built TensorRT engines without needing anything from the export side
(e.g. no `requests`; `transformers`/`torch`/`torchvision` are still required, just for
`Sam3Processor`/`AutoTokenizer` and tensor plumbing around the TRT calls, not for downloading the
full PyTorch model weights). Expects the two engines from the export pipeline at
`./onnx_weights/sam3_encoder_fp16.engine` and
`./onnx_weights/sam3_decoder_batched_fp16.engine` relative to the working directory (`ENCODER_TRT`
/ `DECODER_BATCH_TRT` constants in `mask_extraction_node.py` — change if your engines live
elsewhere). `batch_process_bags.py`'s `--bags-dir` and `IMAGE_FOLDER` in `body_segmentation.py`
default to this project's own local paths — override them for your own data.
