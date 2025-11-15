import onnx
from pathlib import Path

def get_onnx_type_name(elem_type):
    type_map = {
        1: "TYPE_FLOAT",
        2: "TYPE_UINT8",
        3: "TYPE_INT8",
        4: "TYPE_UINT16",
        5: "TYPE_INT16",
        6: "TYPE_INT32",
        7: "TYPE_INT64",
        8: "TYPE_STRING",
        9: "TYPE_BOOL",
        10: "TYPE_FLOAT16",
        11: "TYPE_DOUBLE",
        12: "TYPE_UINT32",
        13: "TYPE_UINT64",
    }
    return type_map.get(elem_type, f"UNKNOWN({elem_type})")

def inspect_onnx_model(model_path):
    if not Path(model_path).exists():
        print(f"Error: {model_path} not found")
        return
    
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return
    
    print(f"\n{'='*80}")
    print(f"Model: {model_path}")
    print(f"{'='*80}")
    
    print(f"\nInputs ({len(model.graph.input)}):")
    for idx, input_tensor in enumerate(model.graph.input, 1):
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(f"dynamic({dim.dim_param})")
            else:
                shape.append(-1)
        
        elem_type = input_tensor.type.tensor_type.elem_type
        print(f"  [{idx}] Name: {input_tensor.name}")
        print(f"      Shape: {shape}")
        print(f"      Dtype: {get_onnx_type_name(elem_type)} ({elem_type})")
    
    print(f"\nOutputs ({len(model.graph.output)}):")
    for idx, output_tensor in enumerate(model.graph.output, 1):
        shape = []
        for dim in output_tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(f"dynamic({dim.dim_param})")
            else:
                shape.append(-1)
        
        elem_type = output_tensor.type.tensor_type.elem_type
        print(f"  [{idx}] Name: {output_tensor.name}")
        print(f"      Shape: {shape}")
        print(f"      Dtype: {get_onnx_type_name(elem_type)} ({elem_type})")
    
    print(f"\n{'='*80}\n")

def main():
    base_path = Path(__file__).parent.parent
    model_dir = base_path / "model"
    
    models = [
        model_dir / "vision_encoder.onnx",
        model_dir / "qformer.onnx",
        model_dir / "opt_decoder_with_past.onnx",
    ]
    
    for model_path in models:
        inspect_onnx_model(str(model_path))

if __name__ == "__main__":
    main()

