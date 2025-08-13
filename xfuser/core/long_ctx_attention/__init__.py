try:
    from .hybrid import (
        xFuserLongContextAttention, 
        xFuserSanaLinearLongContextAttention,
        AttnType,)
    __all__ = [
        "xFuserLongContextAttention",
        "xFuserSanaLinearLongContextAttention",
        "AttnType",
    ]
except ImportError:
    # yunchang not available on Intel GPUs
    __all__ = []
