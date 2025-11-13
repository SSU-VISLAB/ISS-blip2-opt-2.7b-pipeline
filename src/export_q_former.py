import torch

def export_q_former(model):
    qformer = model.qformer.eval()
    vision_embeds = torch.randn(1, 257, 1408).to(qformer.device)  # BLIP2 기본 shape
    attention_mask = torch.ones(1, 257).to(qformer.device)
    query_tokens = model.query_tokens
    torch.onnx.export(
        qformer,
        (vision_embeds, attention_mask, query_tokens),
        "qformer.onnx",
        input_names=["vision_embeds", "attention_mask", "query_tokens"],
        output_names=["qformer_output"],
        dynamic_axes={
            "vision_embeds": {0: "batch"},
            "attention_mask": {0: "batch"},
            "qformer_output": {0: "batch"},
        },
        opset_version=17,
    )