import onnx
from onnx import shape_inference
model = onnx.load("deeplabv3_mnv2_ade20k.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "deeplabv3_mnv2_ade20k_inferred.onnx")

model = onnx.load("mobilebert.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "mobilebert_inferred.onnx")

model = onnx.load("mobilenet_edgetpu.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "mobilenet_edgetpu_inferred.onnx")

model = onnx.load("ssd_mobilenet_v2.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "ssd_mobilenet_v2_inferred.onnx")
