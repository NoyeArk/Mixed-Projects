from policy_value_net import PolicyValueNet

import torch.onnx


# 转为ONNX
def Convert_ONNX(model):
    # 设置模型为推理模式
    model.eval()

    # 设置模型输入的尺寸
    dummy_input = torch.randn(2, 11, 11, requires_grad=True)

    # 导出ONNX模型
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "xxx.onnx",  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})

    print(" ")
    print('Model has been converted to ONNX')


if __name__ == "__main__":
    # 构建模型并训练
    # xxxxxxxxxxxx

    # 测试模型精度
    # testAccuracy()

    # 加载模型结构与权重
    model = PolicyValueNet()

    checkpoint = torch.load('20b128c_hex11.pth')  # 加载检查点文件

    # 使用 load_state_dict 来加载模型权重
    model.load_state_dict(checkpoint)

    # model.load_state_dict(torch.load(path))

    # 转换为ONNX
    Convert_ONNX(model)