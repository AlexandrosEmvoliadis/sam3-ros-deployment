"""
Export SAM-3 as two ONNX models:
  - sam3_encoder.onnx: pixel_values -> 9 vision embedding tensors (run once per image)
  - sam3_decoder.onnx: vision embeddings + text tokens -> masks/boxes/logits (run per prompt)

This gives ~2.2x speedup for multi-prompt inference by avoiding redundant encoder passes.
"""
import torch
from pathlib import Path
from transformers.models.sam3 import Sam3Processor, Sam3Model, modeling_sam3
from PIL import Image
import requests

device = "cpu"  # CPU for ONNX export compatibility

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")
model.eval()

# Sample image + prompt for tracing
image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

prompt = "person"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

pixel_values = inputs["pixel_values"]
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Get encoder outputs for decoder tracing
with torch.no_grad():
    vision_outputs = model.vision_encoder(pixel_values)

output_dir = Path("onnx_weights").resolve()
output_dir.mkdir(parents=True, exist_ok=True)


# ===================================================================
# 1. ENCODER WRAPPER
# ===================================================================
class Sam3EncoderWrapper(torch.nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, pixel_values):
        out = self.vision_encoder(pixel_values)
        # Return all 9 tensors: 1 LHS + 4 FPN hidden + 4 FPN pos
        return (
            out.last_hidden_state,          # [1, 5184, 1024]
            out.fpn_hidden_states[0],       # [1, 256, 288, 288]
            out.fpn_hidden_states[1],       # [1, 256, 144, 144]
            out.fpn_hidden_states[2],       # [1, 256, 72, 72]
            out.fpn_hidden_states[3],       # [1, 256, 36, 36]
            out.fpn_position_encoding[0],   # [1, 256, 288, 288]
            out.fpn_position_encoding[1],   # [1, 256, 144, 144]
            out.fpn_position_encoding[2],   # [1, 256, 72, 72]
            out.fpn_position_encoding[3],   # [1, 256, 36, 36]
        )


encoder_wrapper = Sam3EncoderWrapper(model.vision_encoder).to(device).eval()

encoder_path = str(output_dir / "sam3_encoder.onnx")
print("Exporting encoder...")
print(f"  pixel_values: {pixel_values.shape}")

torch.onnx.export(
    encoder_wrapper,
    (pixel_values,),
    encoder_path,
    input_names=["pixel_values"],
    output_names=[
        "last_hidden_state",
        "fpn_hidden_0", "fpn_hidden_1", "fpn_hidden_2", "fpn_hidden_3",
        "fpn_pos_0", "fpn_pos_1", "fpn_pos_2", "fpn_pos_3",
    ],
    dynamo=False,
    opset_version=17,
    use_external_data_format=True,
)
print(f"  -> {encoder_path}")


# ===================================================================
# 2. DECODER WRAPPER
# ===================================================================
class Sam3DecoderWrapper(torch.nn.Module):
    def __init__(self, sam3_model):
        super().__init__()
        self.sam3 = sam3_model

    def forward(
        self,
        last_hidden_state,
        fpn_hidden_0, fpn_hidden_1, fpn_hidden_2, fpn_hidden_3,
        fpn_pos_0, fpn_pos_1, fpn_pos_2, fpn_pos_3,
        input_ids, attention_mask,
    ):
        # Reconstruct the Sam3VisionEncoderOutput
        vision_embeds = modeling_sam3.Sam3VisionEncoderOutput(
            last_hidden_state=last_hidden_state,
            fpn_hidden_states=(fpn_hidden_0, fpn_hidden_1, fpn_hidden_2, fpn_hidden_3),
            fpn_position_encoding=(fpn_pos_0, fpn_pos_1, fpn_pos_2, fpn_pos_3),
        )

        outputs = self.sam3(
            vision_embeds=vision_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return (
            outputs.pred_masks,         # [1, 200, 288, 288]
            outputs.pred_logits,        # [1, 200]
            outputs.pred_boxes,         # [1, 200, 4]
            outputs.semantic_seg,       # [1, 1, 288, 288]
            outputs.presence_logits,    # [1, 1]
        )


decoder_wrapper = Sam3DecoderWrapper(model).to(device).eval()

# Prepare decoder trace inputs (encoder outputs + text tokens)
lhs = vision_outputs.last_hidden_state
fpn_h = list(vision_outputs.fpn_hidden_states)
fpn_p = list(vision_outputs.fpn_position_encoding)

decoder_inputs = (
    lhs,
    fpn_h[0], fpn_h[1], fpn_h[2], fpn_h[3],
    fpn_p[0], fpn_p[1], fpn_p[2], fpn_p[3],
    input_ids, attention_mask,
)

print(f"\nDecoder trace input shapes:")
names = ["last_hidden_state",
         "fpn_hidden_0", "fpn_hidden_1", "fpn_hidden_2", "fpn_hidden_3",
         "fpn_pos_0", "fpn_pos_1", "fpn_pos_2", "fpn_pos_3",
         "input_ids", "attention_mask"]
for n, t in zip(names, decoder_inputs):
    print(f"  {n}: {t.shape} {t.dtype}")

decoder_path = str(output_dir / "sam3_decoder.onnx")
print("\nExporting decoder...")

# Only text tokens have dynamic sequence length
dynamic_axes = {
    "input_ids":      {1: "sequence_length"},
    "attention_mask": {1: "sequence_length"},
}

torch.onnx.export(
    decoder_wrapper,
    decoder_inputs,
    decoder_path,
    input_names=[
        "last_hidden_state",
        "fpn_hidden_0", "fpn_hidden_1", "fpn_hidden_2", "fpn_hidden_3",
        "fpn_pos_0", "fpn_pos_1", "fpn_pos_2", "fpn_pos_3",
        "input_ids", "attention_mask",
    ],
    output_names=["pred_masks", "pred_logits", "pred_boxes", "semantic_seg", "presence_logits"],
    dynamic_axes=dynamic_axes,
    dynamo=False,
    opset_version=17,
    use_external_data_format=True,
)
print(f"  -> {decoder_path}")

print(f"\nDone. Export summary:")
print(f"  Encoder: {encoder_path}")
print(f"  Decoder: {decoder_path}")
print(f"  Usage: run encoder once per image, decoder once per prompt")