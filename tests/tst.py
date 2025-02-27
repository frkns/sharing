import torch
import torch.nn.functional as F
import time

# Parameters
n_iter = 10000  # Number of iterations for benchmarking
tensor_size = (1000000,)  # 1M elements to see meaningful differences

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tensors
Q_pred = torch.randn(tensor_size, device=device, requires_grad=True)
y = torch.randn(tensor_size, device=device)

# Warm-up runs (to initialize CUDA context if needed)
for _ in range(100):
    _ = (Q_pred - y).pow(2).mean()
    _ = F.mse_loss(Q_pred, y)

def benchmark(fn, name):
    if device.type == "cuda":
        torch.cuda.synchronize()  # Wait for all CUDA ops to finish
    start = time.time()
    
    for _ in range(n_iter):
        loss = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()  # Sync after each iteration
    
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.6f} seconds")
    return elapsed

# Benchmark manual MSE
manual_time = benchmark(
    lambda: (Q_pred - y).pow(2).mean(),
    "Manual MSE"
)

# Benchmark built-in MSE
builtin_time = benchmark(
    lambda: F.mse_loss(Q_pred, y),
    "Built-in MSE"
)

print(f"\nSpeedup factor: {manual_time / builtin_time:.2f}x")
