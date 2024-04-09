#!/bin/bash

# get from here: https://github.com/onnx/tensorflow-onnx
python -m tf2onnx.convert --opset 16 --tflite deeplabv3_mnv2_ade20k_float.tflite --output deeplabv3_mnv2_ade20k.onnx
python -m tf2onnx.convert --opset 16 --tflite mobilebert_float_384_20200602.tflite  --output mobilebert.onnx
python -m tf2onnx.convert --opset 16 --tflite mobilenet_edgetpu_224_1.0_float.tflite --output mobilenet_edgetpu.onnx
python -m tf2onnx.convert --opset 16 --tflite ssd_mobilenet_v2_300_float.tflite --output ssd_mobilenet_v2.onnx
