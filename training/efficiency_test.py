# Test efficiency metrics: Params, MACs, inference time, and saved model size

import torch
import time
import os
from pathlib import Path
from thop import profile
from ptflops import get_model_complexity_info
from model import EfficientNetModel

def test_model_efficiency(model, input_size=(3, 224, 224)):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_input = torch.randn(1, *input_size).to(device)
    model = model.to(device)
    model.eval()

    output_dir = Path("results/efficiency_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "efficiency_metrics.txt"
    log_lines = []

    line = "🔍 Measuring with THOP..."
    print(line)
    log_lines.append(line)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    line = f"MACs: {macs / 1e6:.2f} M"
    print(line)
    log_lines.append(line)
    line = f"Params: {params / 1e6:.2f} M"
    print(line)
    log_lines.append(line)

    line = "\n🔍 Measuring with ptflops..."
    print(line)
    log_lines.append(line)
    with torch.amp.autocast(device_type='cpu', enabled=False):
        macs2, params2 = get_model_complexity_info(
            model,
            input_res=(3, 224, 224),
            as_strings=True,
            input_constructor=lambda: {'x': torch.randn(1, 3, 224, 224)},
            print_per_layer_stat=False,
            verbose=False
        )
    line = f"ptflops MACs: {macs2}"
    print(line)
    log_lines.append(line)
    line = f"ptflops Params: {params2}"
    print(line)
    log_lines.append(line)

    line = "\n🚀 Measuring Inference Time..."
    print(line)
    log_lines.append(line)
    n_runs = 100
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)
    end = time.time()
    line = f"Avg Inference Time: {(end - start) / n_runs * 1000:.2f} ms"
    print(line)
    log_lines.append(line)

    line = "\n📦 Measuring Model Size..."
    print(line)
    log_lines.append(line)
    torch.save(model.state_dict(), "temp_weights.pth")
    size_mb = os.path.getsize("temp_weights.pth") / (1024 * 1024)
    line = f"State Dict Size: {size_mb:.2f} MB"
    print(line)
    log_lines.append(line)
    os.remove("temp_weights.pth")

    with open(log_path, "w") as f:
        for line in log_lines:
            f.write(line + "\n")
    print(f"\n📁 Results saved to: {log_path.resolve()}")

if __name__ == "__main__":
    # Initialize model and get the PyTorch model instance
    model_wrapper = EfficientNetModel()
    model = model_wrapper.get_model()
    test_model_efficiency(model)
