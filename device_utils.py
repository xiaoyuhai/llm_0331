"""设备选择工具：cuda > mps > cpu。"""

from __future__ import annotations

import torch


class DeviceHelper:
    """根据当前环境选择最合适的 torch 设备。"""

    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @classmethod
    def resolve(cls, device: str | torch.device | None = None) -> torch.device:
        """传入 device 则规范化；传 None 则自动选择。"""
        if device is None:
            return cls.get_device()
        return torch.device(device)

    @classmethod
    def print_device(cls, device: str | torch.device | None = None) -> torch.device:
        resolved = cls.resolve(device)
        print(f"using device: {resolved}")
        return resolved


# 兼容旧写法：from device_utils import get_device
def get_device() -> torch.device:
    return DeviceHelper.get_device()
