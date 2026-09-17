"""
Build TRT engine for batched decoder.
Encoder engine stays the same (batch=1).
"""
import os, sys
import tensorrt as trt

DECODER_ONNX = "./onnx_weights/sam3_decoder_batched.onnx"
DECODER_ENGINE = "./onnx_weights/sam3_decoder_batched_fp16.engine"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_batched_decoder_engine():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))

    config.set_tactic_sources(
        1 << int(trt.TacticSource.CUBLAS) |
        1 << int(trt.TacticSource.CUBLAS_LT) |
        1 << int(trt.TacticSource.CUDNN)
    )

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    print(f"Parsing {DECODER_ONNX}...")
    with open(DECODER_ONNX, "rb") as f:
        if not parser.parse(f.read(), os.path.abspath(DECODER_ONNX)):
            for i in range(parser.num_errors):
                print(f"  Error: {parser.get_error(i)}")
            sys.exit(1)

    profile = builder.create_optimization_profile()

    # Vision embedding inputs: dynamic batch (1-8), static spatial dims
    # Shapes from diagnostic: lhs=[N,5184,1024], fpn=[N,256,H,W]
    for name, spatial in [
        ("last_hidden_state", (5184, 1024)),
        ("fpn_hidden_0", (256, 288, 288)),
        ("fpn_hidden_1", (256, 144, 144)),
        ("fpn_hidden_2", (256, 72, 72)),
        ("fpn_hidden_3", (256, 36, 36)),
        ("fpn_pos_0", (256, 288, 288)),
        ("fpn_pos_1", (256, 144, 144)),
        ("fpn_pos_2", (256, 72, 72)),
        ("fpn_pos_3", (256, 36, 36)),
    ]:
        min_s = (1, *spatial)
        opt_s = (5, *spatial)   # optimize for 5 prompts (your PPE count)
        max_s = (8, *spatial)   # allow up to 8
        profile.set_shape(name, min=min_s, opt=opt_s, max=max_s)

    # Text inputs: dynamic batch + dynamic sequence length
    profile.set_shape("input_ids",      min=(1, 1), opt=(5, 32), max=(8, 77))
    profile.set_shape("attention_mask",  min=(1, 1), opt=(5, 32), max=(8, 77))

    config.add_optimization_profile(profile)

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  IN:  {inp.name} {inp.shape}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  OUT: {out.name} {out.shape}")

    print("Building engine (may take 15-30 min)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: build failed")
        sys.exit(1)

    with open(DECODER_ENGINE, "wb") as f:
        f.write(serialized)
    print(f"Saved: {DECODER_ENGINE} ({os.path.getsize(DECODER_ENGINE)/1e6:.1f} MB)")


if __name__ == "__main__":
    build_batched_decoder_engine()