try:
    from .attn_layer import (
        xFuserLongContextAttention,
        xFuserSanaLinearLongContextAttention,
        AttnType,
    )
    __all__ = [
        "xFuserLongContextAttention",
        "xFuserSanaLinearLongContextAttention",
        "AttnType",
    ]
except (ImportError, IndexError):
    # yunchang not available on Intel GPUs
    __all__ = []
