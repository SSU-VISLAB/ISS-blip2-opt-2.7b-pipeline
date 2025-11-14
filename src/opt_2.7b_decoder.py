import torch


def export_opt_decoder(model):
    decoder = model.language_model.eval()
    input_ids = torch.tensor([[1, 2, 3]]).to(decoder.device) # dummy token
    attention_mask = torch.ones_like(input_ids).to(decoder.device)

    class OPTDecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, input_ids, attention_mask):
            outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

            return outputs.logits

    wrapper = OPTDecoderWrapper(decoder)

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        "model/opt_decoder_with_past.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=18,
    )