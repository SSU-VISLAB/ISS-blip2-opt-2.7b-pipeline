#!/bin/bash

set -e

MODEL_DIR="/workspace/model"
OUTPUT_DIR="/workspace/output"

mkdir -p ${OUTPUT_DIR}

echo "Converting Vision Encoder..."
trtexec --onnx=${MODEL_DIR}/vision_encoder.onnx \
        --saveEngine=${OUTPUT_DIR}/vision_encoder.engine \
        --stronglyTyped

echo "Converting Q-Former..."
trtexec --onnx=${MODEL_DIR}/qformer.onnx \
        --saveEngine=${OUTPUT_DIR}/qformer.engine \
        --stronglyTyped

echo "Converting OPT Decoder..."
trtexec --onnx=${MODEL_DIR}/opt_decoder_with_past.onnx \
        --saveEngine=${OUTPUT_DIR}/opt_decoder_with_past.engine \
        --stronglyTyped \
        --minShapes=input_ids:1x1,attention_mask:1x1 \
        --optShapes=input_ids:1x10,attention_mask:1x10 \
        --maxShapes=input_ids:1x50,attention_mask:1x50

echo "All models converted successfully!"

