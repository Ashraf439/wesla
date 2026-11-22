"""WESLA: Web-Enhanced Self-Learning Agent with Adaptive Memory and Selective Continual Learning."""
from .agent import WESLA
from .config import Config, load_config

__all__ = ["WESLA", "Config", "load_config"]
__version__ = "0.1.0"
