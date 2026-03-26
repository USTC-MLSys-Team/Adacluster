import os

# 检查环境变量
use_flashinfer = os.environ.get("USE_FLASHINFER", "0") == "1"

if use_flashinfer:
    print("[INFO] Using FlashInfer implementation")
    # 修改 __init__.py 导入
    from . import model_flashinfer
    # 替换 WanModel
    model.WanModel = model_flashinfer.WanModel
else:
    print("[INFO] Using Triton implementation")
    # 使用默认实现
