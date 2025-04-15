import torch
import torch.nn.functional as F
from typing import Optional

def selective_scan_ref(
    x: torch.Tensor,
    delta: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    Cs: torch.Tensor,
    Ds: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    参考实现：对输入张量 x 沿时间（或序列）维度进行前缀扫描操作，
    并结合 delta、As、Bs、Cs、Ds 等参数进行状态更新。

    参数说明：
      x: 输入张量，形状为 (B, ..., L)，其中 L 为序列长度（扫描维度）。
      delta: 与 x 同维度，表示每个时间步的增量项。
      As, Bs, Cs, Ds: 控制状态更新的参数张量（具体含义请参见算法说明）。
      z: 可选参数（例如用于额外激活），默认 None。
      delta_bias: 可选的偏置项，默认 None。
      delta_softplus: 是否对 delta 使用 softplus 激活，默认 False。
      return_last_state: 若为 True，则返回最终状态；否则返回每一步的状态序列。
      **kwargs: 接收其它额外参数（如额外的控制变量）。

    说明：
      此实现为示例版本，仅执行一个简单的累加更新，实际使用时请根据具体需求调整。
    """
    B, *rest, L = x.shape
    # 初始化状态，形状与 x 除序列维度一致
    state = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
    outputs = []
    
    # 将其它参数求和（仅为示例，可以根据实际算法做更复杂的处理）
    extra = (As.sum() + Bs.sum() + Cs.sum() + Ds.sum()) * 0.001

    for t in range(L):
        current = x[..., t]
        current_delta = delta[..., t]
        if delta_softplus:
            current_delta = F.softplus(current_delta)
        if delta_bias is not None:
            current_delta = current_delta + delta_bias
        # 示例更新：state = state + current + delta + extra
        state = state + current + current_delta + extra
        outputs.append(state.unsqueeze(-1))
    output = torch.cat(outputs, dim=-1)
    if return_last_state:
        return state
    else:
        return output

def selective_scan_fn(
    x: torch.Tensor,
    delta: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    Cs: torch.Tensor,
    Ds: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    优化实现：当前直接调用参考实现 selective_scan_ref。
    此函数签名支持 6 个位置参数（x, delta, As, Bs, Cs, Ds）和其它关键字参数，
    与 MedMamba 中的调用保持兼容。

    参数说明同 selective_scan_ref。
    """
    return selective_scan_ref(
        x, delta, As, Bs, Cs, Ds,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
        **kwargs
    )
