import onnxruntime
import torch
import numpy as np

# 加载ONNX模型
onnx_model = "Lenet.onnx"
session = onnxruntime.InferenceSession(onnx_model)

shape = (1, 2, 11, 11)
dummy_input_np = np.random.randn(*shape).astype(np.float32)

# 准备输入数据（根据模型的输入要求）
input_name = session.get_inputs()[0].name
result = session.run([], {input_name: dummy_input_np})

# 获取模型的输出结果
print("Model output:", result)
