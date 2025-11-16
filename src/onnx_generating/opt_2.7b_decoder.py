import torch


def export_opt_decoder(model):
    decoder = model.language_model.eval()
    language_projection = model.language_projection.eval()
    qformer_output = torch.randn(1, 32, 768).to(decoder.device)
    input_ids = torch.tensor([[1, 2, 3]]).to(decoder.device)
    attention_mask = torch.ones_like(input_ids).to(decoder.device)

    class OPTDecoderWrapper(torch.nn.Module):
        def __init__(self, decoder, language_projection):
            super().__init__()
            self.decoder = decoder
            self.language_projection = language_projection

        def forward(self, qformer_output, input_ids, attention_mask):
            qformer_projected = self.language_projection(qformer_output)
            inputs_embeds = self.decoder.model.decoder.embed_tokens(input_ids)
            inputs_embeds = torch.cat([qformer_projected, inputs_embeds], dim=1)
            
            qformer_attention_mask = torch.ones(qformer_output.shape[0], qformer_output.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
            combined_attention_mask = torch.cat([qformer_attention_mask, attention_mask], dim=1)
            combined_attention_mask = combined_attention_mask.unsqueeze(1).unsqueeze(2)
            combined_attention_mask = combined_attention_mask.to(dtype=torch.float32)
            combined_attention_mask = (1.0 - combined_attention_mask) * -10000.0
            
            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_attention_mask.squeeze(1).squeeze(1),
                use_cache=True,
                return_dict=True,
            )

            return outputs.logits

    wrapper = OPTDecoderWrapper(decoder, language_projection)

    torch.onnx.export(
        wrapper,
        (qformer_output, input_ids, attention_mask),
        "model/opt_decoder_with_past.onnx",
        input_names=["qformer_output", "input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "qformer_output": {0: "batch"},
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=18,
    )