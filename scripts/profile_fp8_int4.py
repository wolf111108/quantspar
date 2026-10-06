"""Profile a real saved Linear workload, or an explicitly synthetic demo."""
import argparse
import tempfile
from pathlib import Path
import torch
from quant import QuantizedLinear,QuantStatManager


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--tensor-file',type=Path,help='torch.save dict: activation, weight, optional bias')
    source.add_argument('--demo',action='store_true',help='synthetic example; not Qwen model statistics')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['prefill','decode','both'],default='both')
    p.add_argument('--nmacro',type=int,default=16)
    p.add_argument('--height',type=int,default=64)
    p.add_argument('--width',type=int,default=48)
    p.add_argument('--banks',type=int,default=16)
    p.add_argument('--as-l',type=int,choices=[0,1],default=1)
    p.add_argument('--bit-scope',choices=['mantissa','sign_mantissa','storage'],default='mantissa')
    p.add_argument('--dynamic-activation',action='store_true')
    p.add_argument('--weight-scale-granularity',choices=['scalar','output_channel'],default='scalar')
    args=p.parse_args(argv)
    if args.output.exists(): raise FileExistsError('Choose a fresh --output path')
    if args.demo:
        torch.manual_seed(0)
        data=dict(activation=torch.randn(1,16,128),weight=torch.randn(48,128),bias=None)
    else:
        data=torch.load(args.tensor_file,map_location='cpu',weights_only=True)
    x=data['activation'].float();w=data['weight'].float();bias=data.get('bias')
    if x.ndim not in (2,3) or w.ndim!=2 or x.shape[-1]!=w.shape[-1]:
        raise ValueError('Expected activation [M,K] or [B,M,K] and weight [N,K]')
    with tempfile.TemporaryDirectory() as scales:
        op=QuantizedLinear(w.shape[1],w.shape[0],bias=bias is not None,
            a_bit='e4m3',w_bit=4,o_bit='none',mixed_precision=False,
            mode='scale_inspection',scale_root_str=scales,
            dynamic_activation=args.dynamic_activation,
            weight_scale_granularity=args.weight_scale_granularity)
        with torch.no_grad():
            op.weight.copy_(w)
            if bias is not None: op.bias.copy_(bias)
            op.set_layer_info('q_proj',0);op(x);op.save_scales();op.mode='quant_forward'
            sm=QuantStatManager(scales,nmacro=args.nmacro,as_l=args.as_l,
                               h=args.height,w=args.width,banks=args.banks,bit_scope=args.bit_scope)
            for phase in (['prefill','decode'] if args.phase=='both' else [args.phase]):
                sm.set_phase(phase)
                inp=x if phase=='prefill' else x[..., :1, :]
                op(inp,stat_collector=sm)
            sm.export_cim_stats(args.output)
    print('Synthetic demo' if args.demo else 'Saved Linear workload')
    for record in sm.cim_records:
        print(f"{record['phase']}: layout={record['layout']}, "
              f"dense={record['dense_steps']}, sparse={record['sparse_steps']}, "
              f"mapped_speedup={record['speedup']}")
    print(args.output)

if __name__=='__main__': main()
