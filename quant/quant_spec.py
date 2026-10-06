"""Fake-quantization formats. quant_awo returns CODES, never dequantized values."""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass(frozen=True)
class QuantSpec:
    kind: str
    bits: Optional[int] = None
    fmt: Optional[str] = None
    enabled: bool = True

    def name(self):
        return f"int{self.bits}" if self.kind == "int" else (self.fmt or "none")


def parse_quant_spec(value, default=8):
    if isinstance(value, QuantSpec):
        return value
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError("Boolean is not a quantization format")
    if isinstance(value, int):
        int_qmax(value)
        return QuantSpec("int", bits=value)
    v = str(value).lower().strip()
    if v.isdigit() or (v.startswith("int") and v[3:].isdigit()):
        return parse_quant_spec(int(v.removeprefix("int")))
    aliases = {"fp8":"e4m3", "e4m3fn":"e4m3", "fp8_e4m3":"e4m3",
               "fp8_e4m3fn":"e4m3", "float8_e4m3":"e4m3", "fp8_e5m2":"e5m2",
               "fp16":"e5m10", "float16":"e5m10", "bfloat16":"bf16",
               "fp4":"e2m1", "fp4_e2m1":"e2m1"}
    v = aliases.get(v,v)
    if v in ("none","disabled","float32","fp32"):
        return QuantSpec("none",enabled=False)
    if v == "bf16":
        return QuantSpec("bf",fmt=v,enabled=False)
    if v in ("e4m3","e5m2","e5m10","e2m1"):
        return QuantSpec("fp",fmt=v,enabled=v not in ("e5m10",))
    raise ValueError(f"Unsupported quant spec: {value}")


def fp8_dtype(fmt):
    fmt = parse_quant_spec(fmt).fmt
    if fmt == "e4m3": return torch.float8_e4m3fn
    if fmt == "e5m2": return torch.float8_e5m2
    raise ValueError(f"Not an FP8 format: {fmt}")


def fp8_max(fmt):
    return float(torch.finfo(fp8_dtype(fmt)).max)


def int_qmax(bits):
    if not isinstance(bits,int) or isinstance(bits,bool) or not 2 <= bits <= 32:
        raise ValueError("Integer width must be in [2,32]")
    return 2 ** (bits-1)


def _range(spec):
    if spec.kind == "int": return int_qmax(spec.bits)-0.5
    if spec.fmt == "e2m1": return 6.0
    return fp8_max(spec.fmt)


def safe_scale_from_tensor(x, spec):
    if not spec.enabled: return 1.0
    if not torch.isfinite(x.float()).all(): raise ValueError("Nonfinite calibration input")
    if x.numel() == 0: return 1.0
    maximum = float(x.detach().float().abs().max())
    return maximum/_range(spec) if maximum > 0 else 1.0


def safe_scale_per_token(x, spec, dim=-1):
    if not torch.isfinite(x.float()).all(): raise ValueError("Nonfinite calibration input")
    shape = list(x.shape); shape[dim]=1
    if not spec.enabled: return torch.ones(shape,device=x.device,dtype=torch.float32)
    maximum=x.detach().float().abs().amax(dim=dim,keepdim=True)
    scale=maximum/_range(spec)
    return torch.where(scale>0,scale,torch.ones_like(scale))


def quant_awo(x, scale, spec, out_dtype=None, chunk_size=1_048_576):
    """Return normalized INT/FP codes in an arithmetic-capable float dtype.

    INT4 grid is [-8,7], scale=max_abs/7.5. Saturation is explicit before FP8
    conversion. Dequantization belongs to the caller (code * scale).
    """
    out_dtype = out_dtype or x.dtype
    if not spec.enabled: return x.to(out_dtype)
    if scale is None or chunk_size <= 0: raise ValueError("Positive scale/chunk_size required")
    scale=torch.as_tensor(scale,device=x.device,dtype=torch.float32)
    if not torch.isfinite(scale).all() or (scale<=0).any(): raise ValueError("Scale must be finite and positive")
    if not torch.isfinite(x.float()).all(): raise ValueError("Nonfinite quantization input")
    values, scales=torch.broadcast_tensors(x.float(),scale)
    flat=values.reshape(-1); scales=scales.reshape(-1)
    result=torch.empty_like(flat)
    for start in range(0,flat.numel(),chunk_size):
        stop=min(flat.numel(),start+chunk_size)
        normalized=flat[start:stop]/scales[start:stop]
        if spec.kind == "int":
            qmax=int_qmax(spec.bits)
            code=normalized.round().clamp(-qmax,qmax-1)
        elif spec.fmt == "e2m1":
            grid=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device=x.device)
            idx=(normalized.abs()[:,None]-grid).abs().argmin(dim=-1)
            code=torch.copysign(grid[idx],normalized)
        elif spec.kind == "fp":
            dtype=fp8_dtype(spec.fmt); maximum=fp8_max(spec.fmt)
            code=normalized.clamp(-maximum,maximum).to(dtype).float()
        else:
            raise ValueError(f"Unsupported enabled format: {spec}")
        result[start:stop]=code
    return result.reshape(values.shape).to(out_dtype)
