#!/bin/bash
echo "Running inference with the generated engine and custom plugin..."
trtexec \
  --loadEngine=output_with_myplugin.engine \
  --staticPlugins=/home/user/Projects/TensorRTStudy/TensorRT/Myplugin/build/libMyPlugin.so \
  --loadInputs=input_ids:bert-base-uncased/trtexec_inputs/input_ids.bin,attention_mask:bert-base-uncased/trtexec_inputs/attention_mask.bin,token_type_ids:bert-base-uncased/trtexec_inputs/token_type_ids.bin \
  --shapes=input_ids:1x16,attention_mask:1x16,token_type_ids:1x16 \
  --dumpOutput \
  --verbose