#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${PROJECT_ROOT}/output"
MODEL_REPOSITORY="${PROJECT_ROOT}/model_repository"

echo "엔진 파일을 Triton model repository로 복사 중..."
echo "출력 디렉터리: ${OUTPUT_DIR}"
echo "모델 저장소: ${MODEL_REPOSITORY}"
echo ""

models=(
    "vision_encoder"
    "qformer"
    "opt_decoder_with_past"
)

for model in "${models[@]}"; do
    engine_file="${OUTPUT_DIR}/${model}.engine"
    target_dir="${MODEL_REPOSITORY}/${model}/1"
    target_file="${target_dir}/${model}.plan"
    
    if [ ! -f "${engine_file}" ]; then
        echo "경고: ${engine_file} 파일을 찾을 수 없습니다. 건너뜁니다."
        continue
    fi
    
    if [ ! -d "${target_dir}" ]; then
        echo "디렉터리 생성: ${target_dir}"
        mkdir -p "${target_dir}"
    fi
    
    echo "복사 중: ${engine_file} -> ${target_file}"
    cp "${engine_file}" "${target_file}"
    echo "완료: ${model}"
    echo ""
done

echo "모든 엔진 파일 복사 완료!"

