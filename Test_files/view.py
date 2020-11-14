import torch

x = torch.randn(4, 4)
print(x.size())  # torch.Size([4, 4])

y = x.view(16)
print(y.size())  # torch.Size([16])


