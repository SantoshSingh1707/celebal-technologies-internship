"""Mini GPT-2 from scratch — package entry points."""
from .config import DataConfig, ModelConfig, TrainConfig
from .model import GPT
from .tokenizer import BPE

__all__ = ["BPE", "GPT", "DataConfig", "ModelConfig", "TrainConfig"]
__version__ = "1.0.0"
