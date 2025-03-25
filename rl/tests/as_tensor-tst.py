import torch
import numpy as np
import time

arr = np.random.rand(1000000)

# Using as_tensor
start = time.time()
t1 = torch.as_tensor(arr)
end = time.time()
print(f"as_tensor: {end - start:.6f} sec")

# Using tensor
start = time.time()
t2 = torch.tensor(arr)
end = time.time()
print(f"tensor: {end - start:.6f} sec")

