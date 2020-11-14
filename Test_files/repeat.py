import torch

x = torch.Tensor([1, 2, 3])

print(x)  # tensor([1., 2., 3.])

print(x.repeat(4, 2))
# tensor([[1., 2., 3., 1., 2., 3.],
#         [1., 2., 3., 1., 2., 3.],
#         [1., 2., 3., 1., 2., 3.],
#         [1., 2., 3., 1., 2., 3.]])



