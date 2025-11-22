# BLIP2-OPT-2.7B ONNX/TensorRT 변환, Trition Serving and demoing with lavis and transformers

## 이력

2025년 11월 16일

- ONNX -> TensorRT 런타임의 모델로 변환을 완료해서 NVIDIA Triton에 서빙을 하는 것은 성공함.
- 이미지를 자연어로 변경은 가능함. 하지만 같은 토큰을 지속적으로 반복해서 출력하는 문제가 있는 것을 확인할 수 있음.

2025년 11월 21일

- ONNX를 거쳐서 TensorRT로 모델을 변환하는 것은 또 하나의 기술적으로 해결해야 하는 요소 중에 하나임.
- 개발 공수가 너무 많이 듦. 공식 채널에서는 `salesforce-lavis`와 `transformers` 라이브러리를 사용해서 서빙하는 가이드 밖에 제공하지 않음.

2025년 11월 22일

- 주피터 노트북으로 모델을 PoC 해보려고 시도해봄
- lavis, transformers를 사용해서 모델을 로드해서 사용을 해보려고 했으나 우선 모델의 용량이 매우 크다는 점으로 인해서 쉽게 사용할 수 없는 문제가 있음.
- 2022년에 제작된 모델이기 때문에 사용하는 의존성이 명시된 정보가 부족해서 사용하기 까다롭다는 판단.
- 제대로 사용을 해보지 못했음. 서버실 네트워크 속도로 인해서 병목이 생겨서 작업을 할 수 없는 문제가 있음.
- BLIP2 모델의 사용성은 포기
- 연구 단계에서 현재도 활발히 연구가 이루어지고 있고 보다 사용하기 편한 [FastVLM](https://github.com/apple/ml-fastvlm) 모델로 대체.

## 실행 순서

### 1. 의존성 설치
```bash
uv sync
```

### 2. ONNX 모델 생성
```bash
uv run src/onnx_generating/main.py
```

### 3. TensorRT 엔진 변환
```bash
docker compose up trtexec
```

### 4. 엔진 파일을 Triton Model Repository로 복사
```bash
./scripts/copy_engines_to_repository.sh
```

### 5. Triton Inference Server 실행
```bash
docker compose up triton
```

## 서버 접근

- HTTP API: `http://localhost:8000`
- gRPC API: `localhost:8001`
- Metrics API: `http://localhost:8002`

## 서버 상태 확인

```bash
curl http://localhost:8000/v2/health/ready
```
