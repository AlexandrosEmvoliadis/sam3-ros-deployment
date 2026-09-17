"""
Build TensorRT FP16 engines from split SAM-3 ONNX exports.
  - sam3_encoder_fp16.engine (static shapes)
  - sam3_decoder_fp16.engine (dynamic sequence_length)
"""
import os
import sys
import tensorrt as trt

ENCODER_ONNX = "./onnx_weights/sam3_encoder.onnx"
DECODER_ONNX = "./onnx_weights/sam3_decoder.onnx"
ENCODER_ENGINE = "./onnx_weights/sam3_encoder_fp16.engine"
DECODER_ENGINE = "./onnx_weights/sam3_decoder_fp16.engine"

FP16 = True
WORKSPACE_GB = 4

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_path, engine_path, dynamic_inputs=None, fp16=True, workspace_gb=4):
    """
    Build one TensorRT engine.
    dynamic_inputs: dict of {name: (min_shape, opt_shape, max_shape)} for dynamic axes.
    """
    print(f"\n{'=' * 60}")
    print(f"Building: {onnx_path} -> {engine_path}")
    print(f"{'=' * 60}")
    print(f"TensorRT version: {trt.__version__}")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # Exclude cuTensor to avoid runtime errors
    config.set_tactic_sources(
        1 << int(trt.TacticSource.CUBLAS) |
        1 << int(trt.TacticSource.CUBLAS_LT) |
        1 << int(trt.TacticSource.CUDNN)
    )

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    print(f"Parsing ONNX...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read(), os.path.abspath(onnx_path)):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            sys.exit(1)

    # Optimization profile for dynamic inputs
    if dynamic_inputs:
        profile = builder.create_optimization_profile()
        for name, (min_s, opt_s, max_s) in dynamic_inputs.items():
            profile.set_shape(name, min=min_s, opt=opt_s, max=max_s)
            print(f"  Dynamic: {name} min={min_s} opt={opt_s} max={max_s}")
        config.add_optimization_profile(profile)

    # Print I/O
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  IN:  {inp.name} shape={inp.shape} dtype={inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  OUT: {out.name} shape={out.shape} dtype={out.dtype}")

    print(f"Building engine (this may take a while)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: Engine build failed")
        sys.exit(1)

    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"-> Saved: {engine_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    # 1. Encoder: all static shapes, no dynamic axes
    build_engine(
        ENCODER_ONNX,
        ENCODER_ENGINE,
        dynamic_inputs=None,
        fp16=FP16,
        workspace_gb=WORKSPACE_GB,
    )

    # 2. Decoder: input_ids and attention_mask have dynamic sequence_length
    build_engine(
        DECODER_ONNX,
        DECODER_ENGINE,
        dynamic_inputs={
            "input_ids":      ((1, 1), (1, 32), (1, 77)),
            "attention_mask": ((1, 1), (1, 32), (1, 77)),
        },
        fp16=FP16,
        workspace_gb=WORKSPACE_GB,
    )

    print(f"\nDone. Run benchmark_sam3_v2.py to compare all backends.")