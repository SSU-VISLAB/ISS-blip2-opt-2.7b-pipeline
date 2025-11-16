#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "=================================================================================="
echo "BLIP2-OPT-2.7B 모델 빌드 스크립트"
echo "=================================================================================="
echo ""

# Step 1: ONNX 모델 생성
echo "Step 1/3: ONNX 모델 생성 중..."
echo "--------------------------------------------------------------------------------"
cd "${PROJECT_ROOT}"
uv run src/onnx_generating/main.py
echo "ONNX 모델 생성 완료!"
echo ""

# Step 2: TensorRT 엔진 변환
echo "Step 2/3: TensorRT 엔진 변환 중..."
echo "--------------------------------------------------------------------------------"
echo "Docker 컨테이너를 시작하여 TensorRT 엔진을 생성합니다..."
docker compose up trtexec

# 컨테이너가 완료될 때까지 대기
CONTAINER_NAME="blip2-tensorrt-converter"
echo "컨테이너 완료 대기 중..."
while [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; do
    sleep 2
done

# 컨테이너 종료 상태 확인
EXIT_CODE=$(docker inspect ${CONTAINER_NAME} --format='{{.State.ExitCode}}')
if [ "${EXIT_CODE}" != "0" ]; then
    echo "에러: TensorRT 엔진 변환 실패 (종료 코드: ${EXIT_CODE})"
    echo "컨테이너 로그 확인:"
    docker logs ${CONTAINER_NAME}
    exit 1
fi

echo "TensorRT 엔진 변환 완료!"
echo ""

# Step 3: 엔진 파일을 repository로 복사
echo "Step 3/3: 엔진 파일을 Triton repository로 복사 중..."
echo "--------------------------------------------------------------------------------"
bash "${PROJECT_ROOT}/scripts/copy_engines_to_repository.sh"
echo ""

echo "=================================================================================="
echo "모든 빌드 작업 완료!"
echo "=================================================================================="
echo ""
echo "다음 단계:"
echo "  docker compose up triton  # Triton 서버 시작"

