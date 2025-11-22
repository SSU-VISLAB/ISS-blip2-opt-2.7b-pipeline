# BLIP2-OPT-2.7B ONNX/TensorRT 변환 및 Triton 서빙

## 이력

2025년 11월 16일

- ONNX -> TensorRT 런타임의 모델로 변환을 완료해서 NVIDIA Triton에 서빙을 하는 것은 성공함.
- 이미지를 자연어로 변경은 가능함. 하지만 같은 토큰을 지속적으로 반복해서 출력하는 문제가 있는 것을 확인할 수 있음.

2025년 11월 21일

- ONNX를 거쳐서 TensorRT로 모델을 변환하는 것은 또 하나의 기술적으로 해결해야 하는 요소 중에 하나임.
- 개발 공수가 너무 많이 듦. 공식 채널에서는 `salesforce-lavis`와 `transformers` 라이브러리를 사용해서 서빙하는 가이드 밖에 제공하지 않음.

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
