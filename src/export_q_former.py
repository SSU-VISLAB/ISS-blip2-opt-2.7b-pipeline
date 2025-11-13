import torch

def export_q_former(model):
    qformer = model.qformer.eval()
    vision_embeds = torch.randn(1, 257, 1408).to(qformer.device)  # BLIP2 기본 shape
    vision_attention_mask = torch.ones(vision_embeds.shape[0], vision_embeds.shape[1]).to(qformer.device)
    query_tokens = model.query_tokens.expand(vision_embeds.shape[0], -1, -1).to(qformer.device)
    query_attention_mask = torch.ones(query_tokens.shape[0], query_tokens.shape[1]).to(qformer.device)

    class QFormerWrapper(torch.nn.Module):
        def __init__(self, qformer):
            super().__init__()
            self.qformer = qformer

        def forward(self, vision_embeds, vision_attention_mask, query_attention_mask, query_tokens):
            outputs = self.qformer(
                query_embeds=query_tokens,
                attention_mask=query_attention_mask,
                encoder_hidden_states=vision_embeds,
                encoder_attention_mask=vision_attention_mask,
                return_dict=True,
            )
            return outputs.last_hidden_state

    wrapper = QFormerWrapper(qformer)
    torch.onnx.export(
        wrapper,
        (vision_embeds, vision_attention_mask, query_attention_mask, query_tokens),
        "qformer.onnx",
        input_names=["vision_embeds", "vision_attention_mask", "query_attention_mask", "query_tokens"],
        output_names=["qformer_output"],
        dynamic_axes={
            "vision_embeds": {0: "batch"},
            "vision_attention_mask": {0: "batch"},
            "query_attention_mask": {0: "batch"},
            "qformer_output": {0: "batch"},
        },
        opset_version=18,
    )