from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    REMOTE_MODEL_NAME: str = "Qwen/Qwen3-0.6B"
    REMOTE_MODEL_NAME_BASE: str = "Qwen/Qwen3-0.6B-Base"
    REMOTE_DATASET_NAME: str = "HuggingFaceH4/ultrachat_200k"