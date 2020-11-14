import torch

x = torch.randn(2, 3, 5)
print(x.size())  # torch.Size([2, 3, 5])

print(x.permute(2, 0, 1).size())  # torch.Size([5, 2, 3])

print(torch.FloatTensor().element_size())  # 4

