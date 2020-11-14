import torch

x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
print(x.narrow(0, 0, 2))

# tensor([[1., 2., 3.],
#         [4., 5., 6.]])

# 从1开始索引，缩小的长度为2
print(x.narrow(1, 1, 2))
# tensor([[2., 3.],
#         [5., 6.]])

