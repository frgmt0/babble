"""babble — a tiny from-scratch model that learns to talk from Discord corrections.

Nothing in this package downloads weights, loads a pretrained model, or reads a
corpus. `Babbler(ModelConfig())` is random numbers; everything it ever knows
arrives through the feedback loop in `core.py`.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
