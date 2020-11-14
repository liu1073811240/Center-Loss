import torch

x = torch.zeros(2, 1, 2, 1, 2)
print(x.size())  # torch.Size([2, 1, 2, 1, 2])

y = torch.squeeze(x)
print(y.size())  # torch.Size([2, 2, 2])

y = torch.squeeze(x, 0)
print(y.size())  # torch.Size([2, 1, 2, 1, 2])

y = torch.squeeze(x, 1)
print(y.size())  # torch.Size([2, 2, 1, 2])  指定维度压缩
