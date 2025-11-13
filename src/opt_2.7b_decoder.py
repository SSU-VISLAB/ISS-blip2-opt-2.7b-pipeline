import torch

def export_opt_decoder(model):
    decoder = model.language_model.eval()
    input_ids = torch.tensor([[1, 2, 3]]).to(decoder.device) # dummy token
    attention_mask = torch.ones_like(input_ids).to(decoder.device)
    past = decoder(input_ids, attention_mask).past_key_values
    torch.onnx.export(
        decoder,
        (input_ids[:, -1:], attention_mask[:, -1:], past),
        "opt_decoder_with_past.onnx",
        input_names=["input_ids", "attention_mask"] +
                    [f"past_{i}" for i in range(len(past))],
        output_names=["logits"] + [f"present_{i}" for i in range(len(past))],
        opset_version=17,
        use_external_data_format=True,
    )