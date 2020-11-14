import torch

a = torch.IntTensor([[1, 2, 3], [11, 22, 33]])
b = torch.IntTensor([[4, 5, 6], [44, 55, 66]])

c = torch.stack([a, b], 0)
print(c)
# tensor([[[ 1,  2,  3],
#          [11, 22, 33]],
#
#         [[ 4,  5,  6],
#          [44, 55, 66]]], dtype=torch.int32)

d = torch.stack([a, b], 1)
print(d)
# tensor([[[ 1,  2,  3],
#          [ 4,  5,  6]],
#
#         [[11, 22, 33],
#          [44, 55, 66]]], dtype=torch.int32)

e = torch.stack([a, b], 2)
print(e)
# tensor([[[ 1,  4],
#          [ 2,  5],
#          [ 3,  6]],
#
#         [[11, 44],
#          [22, 55],
#          [33, 66]]], dtype=torch.int32)
