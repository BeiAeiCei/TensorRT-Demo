#!/bin/bash
set -e

echo "[1/2] Build TensorRT base engine..."
trtexec \
  --onnx=bert-base-uncased/model.onnx \
  --minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 \
  --optShapes=input_ids:1x16,attention_mask:1x16,token_type_ids:1x16 \
  --maxShapes=input_ids:1x128,attention_mask:1x128,token_type_ids:1x128 \
  --saveEngine=output_no_plugin.engine \
  --verbose

echo "[2/2] Build TensorRT plugin engine..."
trtexec \
  --onnx=bert-base-uncased/model_add_custom.onnx \
  --staticPlugins=Myplugin/build/libMyPlugin.so \
  --minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 \
  --optShapes=input_ids:1x16,attention_mask:1x16,token_type_ids:1x16 \
  --maxShapes=input_ids:1x128,attention_mask:1x128,token_type_ids:1x128 \
  --saveEngine=output_with_myplugin.engine \
  --verbose