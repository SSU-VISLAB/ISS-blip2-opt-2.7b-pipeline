#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "TensorRT 엔진 메타데이터 확인 중..."
echo "프로젝트 경로: ${PROJECT_ROOT}"
echo ""

docker run --rm --gpus all \
  -v "${PROJECT_ROOT}:/workspace" \
  -w /workspace \
  nvcr.io/nvidia/tensorrt:25.10-py3 \
  python src/inspecting_metadata/inspect_engine_metadata.py

