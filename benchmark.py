import torch
import torch.nn as nn
import time
import numpy as np
from train import SimpleNet

model_fp32 = SimpleNet()
model_fp32.load_state_dict(torch.load('model_fp32.pth'))
model_fp32.eval()

model_int8 = torch.quantization.quantize_dynamic(
    SimpleNet(),
    {nn.Linear},
    dtype=torch.qint8
)

model_int8.load_state_dict(torch.load('model_int8.pth'))
model_int8.eval()

def measure_latency(model, input_tensor, runs=1000):
    latencies = []

    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

        for _ in range(runs):
            start = time.perf_counter()
            output = model(input_tensor)
            end= time.perf_counter()
            latencies.append((end-start)*1000)
    return{
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'p95': np.percentile(latencies, 95),
        'min': np.min(latencies)
    }

dummy_input = torch.randn(1, 1, 28, 28)

print("Benchmarking FP32 model...")
fp32_stats = measure_latency(model_fp32, dummy_input)

print("Benchmarking model...")
int8_stats = measure_latency(model_int8, dummy_input)

print(f"\n{'Metric':<12} {'FP32 (ms)': >12} {'INT8 (ms)':>12}{'Speedup':>10}")
print("-" * 48)

for metric in ['mean', 'median', 'p95', 'min']:
    fp32_val = fp32_stats[metric]
    int8_val = int8_stats[metric]
    speedup = fp32_val/int8_val
    print(f"{metric:<12}{fp32_val:>12.3f}{int8_val:>12.3f}{speedup:>9.2f}")