"""
TensorRT 엔진 메타데이터 확인 스크립트

이 스크립트는 TensorRT 엔진 파일의 입력/출력 정보를 확인합니다.
TensorRT는 CUDA, cuDNN 등 시스템 의존성이 필요하므로 Docker 컨테이너에서 실행해야 합니다.

실행 방법:
    이 스크립트는 직접 실행하지 말고, scripts/inspect_engine_metadata.sh 스크립트를 사용하세요:
    
    ./scripts/inspect_engine_metadata.sh

    또는 Docker 컨테이너에서 직접 실행하려면:
    
    docker run --rm --gpus all \\
      -v /data/jayn2u/iss/blip2-opt-2.7b-onnx:/workspace \\
      -w /workspace \\
      nvcr.io/nvidia/tensorrt:25.10-py3 \\
      python src/inspecting_metadata/inspect_engine_metadata.py

주의사항:
    - TensorRT 엔진 파일은 GPU가 있는 환경에서만 deserialize할 수 있습니다
    - 로컬 환경에서는 CUDA, cuDNN 등이 설치되어 있어야 하므로 Docker 사용을 권장합니다
"""
import tensorrt as trt
from pathlib import Path

def get_dtype_name(dtype):
    dtype_map = {
        trt.float32: "TYPE_FP32",
        trt.float16: "TYPE_FP16",
        trt.int32: "TYPE_INT32",
        trt.int64: "TYPE_INT64",
        trt.int8: "TYPE_INT8",
        trt.uint8: "TYPE_UINT8",
        trt.bool: "TYPE_BOOL",
    }
    return dtype_map.get(dtype, f"UNKNOWN({dtype})")

def inspect_trt_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    
    if not Path(engine_path).exists():
        print(f"Error: {engine_path} not found")
        return
    
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    if engine is None:
        print(f"Error: Failed to deserialize engine from {engine_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"Engine: {engine_path}")
    print(f"{'='*80}")
    
    inputs = []
    outputs = []
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        
        tensor_info = {
            'name': name,
            'shape': shape,
            'dtype': dtype,
            'dtype_name': get_dtype_name(dtype)
        }
        
        if mode == trt.TensorIOMode.INPUT:
            inputs.append(tensor_info)
        elif mode == trt.TensorIOMode.OUTPUT:
            outputs.append(tensor_info)
    
    print(f"\nInputs ({len(inputs)}):")
    for idx, inp in enumerate(inputs, 1):
        print(f"  [{idx}] Name: {inp['name']}")
        print(f"      Shape: {inp['shape']}")
        print(f"      Dtype: {inp['dtype_name']} ({inp['dtype']})")
    
    print(f"\nOutputs ({len(outputs)}):")
    for idx, out in enumerate(outputs, 1):
        print(f"  [{idx}] Name: {out['name']}")
        print(f"      Shape: {out['shape']}")
        print(f"      Dtype: {out['dtype_name']} ({out['dtype']})")
    
    print(f"\n{'='*80}\n")

def main():
    base_path = Path(__file__).parent.parent.parent
    output_dir = base_path / "output"
    
    engines = [
        output_dir / "vision_encoder.engine",
        output_dir / "qformer.engine",
        output_dir / "opt_decoder_with_past.engine",
    ]
    
    for engine_path in engines:
        inspect_trt_engine(str(engine_path))

if __name__ == "__main__":
    main()

