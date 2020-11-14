import torch

x = torch.Tensor([[1], [2], [3]])
print(x.size())
# torch.Size([3, 1])

print(x.expand(3, 4))
# tensor([[1., 1., 1., 1.],
#         [2., 2., 2., 2.],
#         [3., 3., 3., 3.]])

