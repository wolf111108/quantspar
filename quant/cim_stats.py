"""Exact encoded bit counts and analytical Asyn-CIM compute statistics.

The default scope is explicit mantissa only. It excludes sign, hidden one,
exponent alignment and weight-update/IO costs; these are not physical E2E times.
"""
import math
import torch
from .quant_spec import parse_quant_spec, fp8_dtype


def encode_bits(values, spec, scope="mantissa"):
    spec=parse_quant_spec(spec)
    if scope not in ("mantissa","sign_mantissa","storage"):
        raise ValueError("bit scope must be mantissa, sign_mantissa or storage")
    values=values.detach().float()
    if not torch.isfinite(values).all(): raise ValueError("Statistics input must be finite")
    if spec.kind == "int":
        if ((values != values.round()) | (values < -(2**(spec.bits-1))) |
            (values > 2**(spec.bits-1)-1)).any():
            raise ValueError("INT statistics require valid quantized integer codes")
        return values.long() & ((1<<spec.bits)-1), spec.bits
    formats={"e4m3":(3,4,torch.float8_e4m3fn,torch.uint8),
             "e5m2":(2,5,torch.float8_e5m2,torch.uint8),
             "e5m10":(10,5,torch.float16,torch.int16),
             "bf16":(7,8,torch.bfloat16,torch.int16)}
    if spec.fmt == "e2m1":
        grid=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device=values.device)
        distance=(values.abs().unsqueeze(-1)-grid).abs()
        raw=distance.argmin(-1).long() | (values.signbit().long()<<3)
        if (distance.min(-1).values != 0).any(): raise ValueError("FP4 statistics require grid codes")
        mb,eb=1,2
    elif spec.fmt in formats:
        mb,eb,dtype,view=formats[spec.fmt]
        encoded=values.to(dtype)
        if not torch.isfinite(encoded.float()).all(): raise ValueError("Invalid FP encoding")
        # Observations must be codes, not unquantized real-domain activations.
        if spec.enabled and (encoded.float() != values).any():
            raise ValueError("FP statistics require actual normalized quantized codes")
        raw=encoded.contiguous().view(view).long() & ((1<<(mb+eb+1))-1)
    else:
        raise ValueError(f"No encoded bit statistic for {spec}")
    if scope == "storage": return raw,mb+eb+1
    code=raw & ((1<<mb)-1)
    if scope == "sign_mantissa":
        code |= ((raw>>(mb+eb))&1)<<mb
        return code,mb+1
    return code,mb


def popcount(codes,width):
    count=torch.zeros_like(codes,dtype=torch.int16)
    for bit in range(width): count += ((codes>>bit)&1).to(torch.int16)
    return count


def sparse_counts(values,spec,scope="mantissa",threshold=0,chunk_size=1_048_576):
    if chunk_size<=0: raise ValueError("chunk_size must be positive")
    flat=values.detach().reshape(-1);zeros=ones=total_bits=0
    for start in range(0,flat.numel(),chunk_size):
        part=flat[start:start+chunk_size]
        codes,width=encode_bits(part,spec,scope)
        zeros+=int((part.float().abs()<=threshold).sum())
        ones+=int(popcount(codes,width).sum())
        total_bits+=part.numel()*width
    zero_bits=total_bits-ones
    return flat.numel(),zeros,total_bits,zero_bits,zero_bits,zero_bits


def aggregate_bank_steps(bank_steps,asynchronous=True):
    """Input [B,H,Mr,Kr,Mf,Kf,bank]; serialize independent B/H operands."""
    rounds=bank_steps.amax(-1)
    if asynchronous:
        return rounds.sum(dim=(2,3)).amax(dim=(-1,-2)).sum()
    return rounds.amax(dim=(-1,-2)).sum()


def measure_mapping(cim,values,spec,in_features,out_features,is_prefill=True,
                    scope="mantissa",asynchronous=True):
    from .mapping import (padding_x,_effective_h,compute_optimal_macro_layout_prefill,
                          compute_optimal_macro_layout_decode)
    if values.ndim==2: x=values[None,None]
    elif values.ndim==3: x=values[:,None]
    elif values.ndim==4: x=values
    else: raise ValueError("Mapping expects [M,K], [B,M,K] or [B,H,M,K]")
    if x.shape[-1]!=in_features: raise ValueError("Mapping K differs from activation shape")
    fn=compute_optimal_macro_layout_prefill if is_prefill else compute_optimal_macro_layout_decode
    layout=fn(cim.Nmacro,in_features,out_features,x.shape[-2],h=cim.h,w=cim.w,Nadder=cim.Nadder)
    kf,mf,nf,kr,mr,nr=layout
    # Stream token-round chunks and operands. Do not materialize a long-int
    # encoding or an extra bit axis for the whole [B,H,S,K] attention tensor.
    dense_total=sparse_total=0
    chunk_rounds=max(1,256//mf)
    for batch in range(x.shape[0]):
        for head in range(x.shape[1]):
            sparse_lanes=torch.zeros((mf,kf),dtype=torch.int64,device=x.device)
            dense_lanes=torch.zeros_like(sparse_lanes)
            for round_start in range(0,mr,chunk_rounds):
                row_start=round_start*mf
                part=x[batch,head,row_start:min(x.shape[-2],row_start+chunk_rounds*mf)]
                codes,width=encode_bits(part,spec,scope)
                rounds=math.ceil(part.shape[0]/mf)
                chunk_layout=(kf,mf,nf,kr,rounds,nr)
                counts=padding_x(popcount(codes,width)[None,None],in_features,out_features,cim,*chunk_layout)
                valid=padding_x(torch.ones_like(part,dtype=torch.int16)[None,None],
                                in_features,out_features,cim,*chunk_layout)
                sparse_round=counts.sum(-1).amax(-1)[0,0]
                dense_round=(valid.sum(-1)*width).amax(-1)[0,0]
                if asynchronous:
                    sparse_lanes+=sparse_round.sum(dim=(0,1))
                    dense_lanes+=dense_round.sum(dim=(0,1))
                else:
                    sparse_total+=int(sparse_round.amax(dim=(-1,-2)).sum())
                    dense_total+=int(dense_round.amax(dim=(-1,-2)).sum())
            if asynchronous:
                sparse_total+=int(sparse_lanes.max())
                dense_total+=int(dense_lanes.max())
    sparse=sparse_total*nr;dense=dense_total*nr
    # Current LLMCompass uses an analytical capacity denominator. A bridge
    # ratio uses that denominator to recover EXACTLY this sparse cycle total.
    operands=x.shape[0]*x.shape[1]
    reference=_effective_h(cim,in_features)*width*kr*mr*nr*operands
    return dict(layout=list(layout),dense_steps=float(dense),sparse_steps=float(sparse),
                speedup=dense/sparse if sparse>0 else None,
                llmcompass_dense_steps=float(reference),
                llmcompass_speedup=float(reference/sparse) if sparse>0 else None,
                dense_bits=width,bit_scope=scope,
                operand_shape=list(x.shape),in_features=in_features,out_features=out_features,
                aggregation="sum_of_independent_operands_max_of_macro_round_sums" if asynchronous
                            else "sum_of_round_macro_barriers",
                all_counted_bits_zero=bool(sparse==0))


def unit_sparse_counts(values,spec,scope,bit_group_size,dim_group_size,chunk_rows=2048):
    if min(bit_group_size,dim_group_size,chunk_rows)<=0: raise ValueError("Unit grouping must be positive")
    if values.numel()==0: return 0,0
    rows=values.reshape(-1,values.shape[-1]);zeros=total=0
    for start in range(0,rows.shape[0],chunk_rows):
        codes,width=encode_bits(rows[start:start+chunk_rows],spec,scope)
        pad=(-codes.shape[-1])%dim_group_size
        if pad: codes=torch.nn.functional.pad(codes,(0,pad))
        groups=codes.reshape(codes.shape[0],-1,dim_group_size)
        for bit in range(0,width,bit_group_size):
            mask=((1<<min(bit_group_size,width-bit))-1)<<bit
            empty=((groups&mask)==0).all(-1)
            zeros+=int(empty.sum());total+=empty.numel()
    return zeros,total
