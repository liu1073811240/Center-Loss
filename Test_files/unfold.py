import torch

x = torch.arange(1, 8)
print(x)  # tensor([1, 2, 3, 4, 5, 6, 7])

print(x.unfold(0, 2, 1))  # 在该维度上重复元素个数为2，步长为1
# tensor([[1, 2],
#         [2, 3],
#         [3, 4],
#         [4, 5],
#         [5, 6],
#         [6, 7]])

