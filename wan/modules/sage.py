import torch
from typing import Optional
import os

SAGEATTN_AVAILABLE = False
try:
    if os.getenv("DISABLE_SAGEATTENTION", "0") != "0":
        raise Exception("DISABLE_SAGEATTENTION is set")
    
    from sageattention import sageattn

    @torch.library.custom_op("mylib::sageattn", mutates_args={"q", "k", "v"}, device_types="cuda")
    def sageattn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      attn_mask: Optional[torch.Tensor] = None , dropout_p: float = 0, is_causal: bool = False) -> torch.Tensor:
        return sageattn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    
    @sageattn_func.register_fake
    def _sageattn_fake(q, k, v, attn_mask=None, dropout_p=0, is_causal=False):
        return torch.empty(*q.shape, device=q.device, dtype=q.dtype)
    
    print("SageAttention loaded successfully")

    SAGEATTN_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load sageattention: {str(e)}")
    if isinstance(e, ModuleNotFoundError):
        print("sageattention package is not installed")
    elif isinstance(e, ImportError) and "DLL" in str(e):
        print("sageattention DLL loading error")
    sageattn_func = None
