import torch

def export_vision_encoder(model):
    vision = model.vision_model.eval()
    dummy_image = torch.randn(1, 3, 224, 224).to(vision.device)
    torch.onnx.export(
        vision,
        dummy_image,
        "vision_encoder.onnx",
        input_names=["pixel_values"],
        output_names=["vision_embeddings"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "vision_embeddings": {0: "batch"},
        },
        opset_version=18,
    )