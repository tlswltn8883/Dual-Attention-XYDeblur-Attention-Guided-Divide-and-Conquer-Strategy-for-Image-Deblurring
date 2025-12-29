import torch
from fvcore.nn import FlopCountAnalysis
import time
from collections import defaultdict
from models.XYDeblur import XYDeblur

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = XYDeblur().to(device)
model.eval()

dummy_input = torch.randn(1, 3, 256, 256).to(device)

flops = FlopCountAnalysis(model, dummy_input)
flops_g = flops.total() / 1e9

params = sum(p.numel() for p in model.parameters())
params_m = params / 1e6

print(f"FLOPs: {flops_g:.4f} GFLOPs")
print(f"Params: {params_m:.4f} M\n")

torch.cuda.synchronize()
start = time.time()
_ = model(dummy_input)
torch.cuda.synchronize()
end = time.time()

print(f"Inference Time: {(end - start):.4f} s\n")

group_params = defaultdict(int)

for name, module in model.named_modules():

    params = sum(p.numel() for p in module.parameters(recurse=False))

    if params > 0:
        prefix = name.split('.')[0]
        group_params[prefix] += params

for group, total_params in group_params.items():
    print(f"{group:20s}  {total_params / 1e6:.6f} M")
