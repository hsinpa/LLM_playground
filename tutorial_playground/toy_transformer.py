import torch

sequence = 5
dim_k = 3
input_data = torch.rand(sequence, dim_k)

q = torch.clone(input_data)
k = torch.clone(input_data)
v = torch.clone(input_data)

k_t = k.transpose(1, 0)
print(k)
print(k_t)

qk = torch.matmul(q, k_t) # Transpose K for dot operation
#
print(qk.shape, v.shape)

