import onnx

model_name = 'resnet50_v1'

''' save with external data '''
model = onnx.load(f'./{model_name}.onnx')

onnx.save_model(model, f'{model_name}_tmp.onnx', save_as_external_data = True, all_tensors_to_one_file = True, location = f'{model_name}_external_data.onnx', size_threshold = 1024, convert_attribute = False)

''' infer onnx model shape '''
model = onnx.load(f'{model_name}_tmp.onnx')
inferred_model = onnx.shape_inference.infer_shapes(model)
onnx.save(inferred_model, f'{model_name}_inferred_model.onnx')
