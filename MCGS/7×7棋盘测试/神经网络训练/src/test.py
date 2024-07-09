import torch

# 示例多维列表
multi_dimensional_list = [
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]],
    [[13, 14, 15], [16, 17, 18]]
]

# 初始化一个空张量
tensor_list = []

# 遍历多维列表的每个元素并将其转换为张量
for i in range(len(multi_dimensional_list)):
    sub_list = multi_dimensional_list[i]
    sub_tensor_list = []
    for j in range(len(sub_list)):
        sub_tensor = torch.Tensor(sub_list[j])
        sub_tensor_list.append(sub_tensor)
    tensor_list.append(sub_tensor_list)

# 将嵌套列表转换为张量
final_tensor = torch.tensor(tensor_list)

# 打印结果
print(final_tensor)
