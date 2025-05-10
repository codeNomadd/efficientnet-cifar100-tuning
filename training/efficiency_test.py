# Test efficiency metrics: Params, MACs, inference time, and saved model size

import torch
import time
import os
from thop import profile
from ptflops import get_model_complexity_info
from model import build_model  # Make sure model.py has build_model()

def test_model_efficiency(model, input_size=(3, 224, 224)):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_input = torch.randn(1, *input_size).to(device)
    model = model.to(device)
    model.eval()

    print("🔍 Measuring with THOP...")
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"MACs: {macs / 1e6:.2f} M")
    print(f"Params: {params / 1e6:.2f} M")

    print("\n🔍 Measuring with ptflops...")
    with torch.cuda.amp.autocast(enabled=False):
        macs2, params2 = get_model_complexity_info(model, input_res=input_size[1:], as_strings=True,
                                                   print_per_layer_stat=False, verbose=False)
    print(f"ptflops MACs: {macs2}")
    print(f"ptflops Params: {params2}")

    print("\n🚀 Measuring Inference Time...")
    n_runs = 100
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)
    end = time.time()
    print(f"Avg Inference Time: {(end - start) / n_runs * 1000:.2f} ms")

    print("\n📦 Measuring Model Size...")
    torch.save(model.state_dict(), "temp_weights.pth")
    size_mb = os.path.getsize("temp_weights.pth") / (1024 * 1024)
    print(f"State Dict Size: {size_mb:.2f} MB")
    os.remove("temp_weights.pth")

if __name__ == "__main__":
    model = build_model()
    test_model_efficiency(model)
