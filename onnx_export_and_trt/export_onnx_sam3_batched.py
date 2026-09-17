"""
Export batched decoder: vision embeddings [N,...] + text tokens [N, seq_len] → outputs [N,...]
"""
import torch
from pathlib import Path
from transformers.models.sam3 import Sam3Processor, Sam3Model, modeling_sam3
from transformers import AutoTokenizer
from PIL import Image
import requests

device = "cpu"
model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
processor = Sam3Processor.from_pretrained("facebook/sam3")
tokenizer = AutoTokenizer.from_pretrained("facebook/sam3")

image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

queries = ["person", "hard hat", "safety vest"]  # 3 prompts for tracing
N = len(queries)

# Get vision embeddings (batch=1)
inputs_first = processor(images=image, text=queries[0], return_tensors="pt").to(device)
with torch.no_grad():
    vision_embeds = model.vision_encoder(inputs_first["pixel_values"])

# Tokenize all prompts, padded to same length
all_tokens = tokenizer(queries, return_tensors="pt", padding=True).to(device)

# Tile vision embeddings to [N, ...]
lhs = vision_embeds.last_hidden_state.expand(N, -1, -1).contiguous()
fpn_h = [f.expand(N, -1, -1, -1).contiguous() for f in vision_embeds.fpn_hidden_states]
fpn_p = [f.expand(N, -1, -1, -1).contiguous() for f in vision_embeds.fpn_position_encoding]

print("Batched decoder trace inputs:")
print(f"  last_hidden_state: {lhs.shape}")
print(f"  fpn_hidden_0:      {fpn_h[0].shape}")
print(f"  input_ids:         {all_tokens['input_ids'].shape}")
print(f"  attention_mask:    {all_tokens['attention_mask'].shape}")


class Sam3BatchedDecoderWrapper(torch.nn.Module):
    def __init__(self, sam3_model):
        super().__init__()
        self.sam3 = sam3_model

    def forward(self, last_hidden_state,
                fpn_hidden_0, fpn_hidden_1, fpn_hidden_2, fpn_hidden_3,
                fpn_pos_0, fpn_pos_1, fpn_pos_2, fpn_pos_3,
                input_ids, attention_mask):
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
            outputs.pred_masks,
            outputs.pred_logits,
            outputs.pred_boxes,
            outputs.semantic_seg,
            outputs.presence_logits,
        )


wrapper = Sam3BatchedDecoderWrapper(model).to(device).eval()

decoder_inputs = (
    lhs,
    fpn_h[0], fpn_h[1], fpn_h[2], fpn_h[3],
    fpn_p[0], fpn_p[1], fpn_p[2], fpn_p[3],
    all_tokens["input_ids"], all_tokens["attention_mask"],
)

output_dir = Path("onnx_weights").resolve()
decoder_path = str(output_dir / "sam3_decoder_batched.onnx")

# Dynamic: batch (num_prompts) and sequence_length
dynamic_axes = {
    "last_hidden_state": {0: "num_prompts"},
    "fpn_hidden_0": {0: "num_prompts"},
    "fpn_hidden_1": {0: "num_prompts"},
    "fpn_hidden_2": {0: "num_prompts"},
    "fpn_hidden_3": {0: "num_prompts"},
    "fpn_pos_0": {0: "num_prompts"},
    "fpn_pos_1": {0: "num_prompts"},
    "fpn_pos_2": {0: "num_prompts"},
    "fpn_pos_3": {0: "num_prompts"},
    "input_ids": {0: "num_prompts", 1: "sequence_length"},
    "attention_mask": {0: "num_prompts", 1: "sequence_length"},
    "pred_masks": {0: "num_prompts"},
    "pred_logits": {0: "num_prompts"},
    "pred_boxes": {0: "num_prompts"},
    "semantic_seg": {0: "num_prompts"},
    "presence_logits": {0: "num_prompts"},
}

print(f"\nExporting batched decoder to {decoder_path}...")
torch.onnx.export(
    wrapper,
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
print("Done.")