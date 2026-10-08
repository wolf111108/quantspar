"""Real CPU PyTorch checks, independent of model checkpoints or import shims."""
import json
import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from quant import QuantizedLinear, QuantizedMatMul, QuantStatManager
from quant.quant_spec import parse_quant_spec, quant_awo
from quant.cim_stats import sparse_counts, measure_mapping
from quant.mapping import CIM_sys, padding_x, _to_fp8_e4m3_bits, Mapping_stat_dynamic


class FP8INT4Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)

    def linear(self,dynamic=False,granularity='scalar',output='none'):
        op=QuantizedLinear(128,7,a_bit='e4m3',w_bit=4,o_bit=output,
                           scale_root_str=self.tmp.name,mode='scale_inspection',
                           dynamic_activation=dynamic,weight_scale_granularity=granularity)
        op.set_layer_info('q_proj',0)
        return op

    def calibrate(self,op,*xs):
        op(*xs);op.save_scales();op.mode='quant_forward'

    def test_default_is_fixed_fp8_not_mixed_precision(self):
        self.assertFalse(self.linear().mixed_precision)
        self.assertFalse(QuantizedMatMul(A_bit='e4m3',B_bit='e4m3',O_bit='none').mixed_precision)
        self.assertEqual(parse_quant_spec('fp8').fmt,'e4m3')
        self.assertEqual(parse_quant_spec('int4').bits,4)

    def test_fp8_saturation_int4_bounds_and_chunk_equivalence(self):
        x=torch.tensor([-1000.,-8.,-1.,0.,1.,7.,1000.])
        q4=quant_awo(x,1,parse_quant_spec(4),out_dtype=torch.float32,chunk_size=2)
        self.assertEqual(q4.tolist(),[-8.,-8.,-1.,0.,1.,7.,7.])
        q8=quant_awo(x,1,parse_quant_spec('e4m3'),out_dtype=torch.float32,chunk_size=2)
        self.assertTrue(torch.isfinite(q8).all());self.assertEqual(float(q8.abs().max()),448)
        torch.testing.assert_close(q8,quant_awo(x,1,parse_quant_spec('e4m3'),out_dtype=torch.float32))
        with self.assertRaises(ValueError): quant_awo(x,0,parse_quant_spec(4))
        with self.assertRaises(ValueError): quant_awo(torch.tensor([float('nan')]),1,parse_quant_spec('e4m3'))

    def test_disabled_output_preserves_quantized_operands_and_bias(self):
        op=self.linear();x=torch.randn(2,5,128);self.calibrate(op,x)
        got=op(x)
        a=quant_awo(x,op.a_interval,op.a_spec,out_dtype=torch.float32)*op.a_interval
        w=quant_awo(op.weight,op.w_interval,op.w_spec,out_dtype=torch.float32)*op.w_interval
        ref=F.linear(a,w,op.bias)
        torch.testing.assert_close(got,ref)
        self.assertGreater(float((got-F.linear(x,op.weight,op.bias)).abs().max()),1e-3)

    def test_dynamic_and_channelwise_weights_numerical_reference(self):
        for granularity in ['scalar','output_channel']:
            for dynamic in [False,True]:
                with self.subTest(granularity=granularity,dynamic=dynamic):
                    op=self.linear(dynamic,granularity);x=torch.randn(2,5,128)
                    self.calibrate(op,x)
                    sm=QuantStatManager(self.tmp.name,nmacro=4);sm.set_phase('prefill')
                    got=op(x,stat_collector=sm)
                    scale=(x.abs().amax(-1,keepdim=True)/448 if dynamic else op.a_interval)
                    a=(x/scale).clamp(-448,448).to(torch.float8_e4m3fn).float()*scale
                    ws=torch.as_tensor(op.w_interval)
                    if ws.ndim: ws=ws[:,None]
                    w=(op.weight/ws).round().clamp(-8,7)*ws
                    torch.testing.assert_close(got,F.linear(a,w,op.bias))
                    self.assertEqual(len(sm.cim_records),1)
                    self.assertEqual(sm.cim_records[0]['operand_shape'],[1,1,10,128])

    def test_fp8_output_is_applied_to_quantized_result(self):
        op=self.linear(output='e4m3');x=torch.randn(1,3,128);self.calibrate(op,x)
        a=quant_awo(x,op.a_interval,op.a_spec,out_dtype=torch.float32)*op.a_interval
        w=quant_awo(op.weight,op.w_interval,op.w_spec,out_dtype=torch.float32)*op.w_interval
        real=F.linear(a,w,op.bias)
        ref=(real/op.o_interval).clamp(-448,448).to(torch.float8_e4m3fn).float()*op.o_interval
        torch.testing.assert_close(op(x),ref)

    def test_matmul_fp8_operands_and_unquantized_output(self):
        op=QuantizedMatMul(A_bit='e4m3',B_bit='e4m3',O_bit='none',
                          scale_root_str=self.tmp.name,mode='scale_inspection')
        op.set_layer_info('qk_matmul',0)
        a=torch.randn(2,3,4,128);b=torch.randn(2,3,128,9)
        self.calibrate(op,a,b)
        sm=QuantStatManager(self.tmp.name,nmacro=4);sm.set_phase('decode')
        got=op(a,b,stat_collector=sm)
        ac=(a/op.A_interval).clamp(-448,448).to(torch.float8_e4m3fn).float()*op.A_interval
        bc=(b/op.B_interval).clamp(-448,448).to(torch.float8_e4m3fn).float()*op.B_interval
        torch.testing.assert_close(got,ac@bc)
        self.assertEqual(sm.cim_records[0]['operand_shape'],[2,3,4,128])
        self.assertEqual(sm.dynamic_weight_element_count,b.numel())

    def test_exact_fp8_element_and_bit_counts(self):
        codes=torch.tensor([0.,1.,1.875,-1.5,2**-9,-2**-9])
        self.assertEqual(sparse_counts(codes,'e4m3',chunk_size=2),(6,1,18,12,12,12))
        self.assertEqual(sparse_counts(codes,'e4m3','sign_mantissa'),(6,1,24,16,16,16))
        self.assertEqual(sparse_counts(torch.empty(0),'e4m3'),(0,0,0,0,0,0))
        with self.assertRaises(ValueError): sparse_counts(torch.tensor([1.1]),'e4m3')

    def test_int4_storage_counts_include_negative_eight(self):
        self.assertEqual(sparse_counts(torch.tensor([-8.,-1.,0.,7.]),4),(4,1,16,8,8,8))
        with self.assertRaises(ValueError): sparse_counts(torch.tensor([8.]),4)

    def test_native_fp8_raw_bits_at_subnormal_boundary(self):
        x=torch.tensor([0.,-0.,2**-9,7.5*2**-9,2**-6,1.0625,448.,500.])
        _,mantissa,raw=_to_fp8_e4m3_bits(x)
        native=x.clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8)
        torch.testing.assert_close(raw,native);torch.testing.assert_close(mantissa,raw&7)

    def test_every_counted_bit_active_is_one_x_even_for_small_k_and_tails(self):
        for k in [16,128,1024,1025,4096]:
            for prefill in [True,False]:
                x=torch.full((5,k),1.875)
                result=measure_mapping(CIM_sys(64,48,16,16,int(1e9)),x,'e4m3',k,137,prefill)
                self.assertEqual(result['speedup'],1)
                self.assertEqual(result['sparse_steps'],result['dense_steps'])
        small=measure_mapping(CIM_sys(64,48,16,16,int(1e9)),torch.full((16,128),1.875),'e4m3',128,4096)
        self.assertEqual(small['sparse_steps'],2064)
        self.assertEqual(small['dense_steps'],2064)

    def test_padding_preserves_token_k_slices(self):
        cim=CIM_sys(h=2,w=2,Nadder=2,Nmacro=4,freq=int(1e9))
        x=torch.arange(4*9).reshape(1,1,4,9)
        # 2 token lanes, 2 K lanes, 2 K rounds, 2 token rounds.
        actual=padding_x(x,9,2,cim,2,2,1,2,2,1)
        for mr in range(2):
            for kr in range(2):
                for mf in range(2):
                    for kf in range(2):
                        for bank in range(2):
                            for h in range(2):
                                pos=((kr*2+kf)*2+bank)*2+h
                                expected=x[0,0,mr*2+mf,pos] if pos<9 else 0
                                self.assertEqual(actual[0,0,mr,kr,mf,kf,bank,h],expected)

    def test_mapping_matches_independent_scalar_scheduler(self):
        cim=CIM_sys(h=2,w=2,Nadder=2,Nmacro=4,freq=int(1e9))
        x=torch.tensor([[[1+((i+j)%8)/8 for j in range(9)] for i in range(5)],
                        [[1+((2*i+j)%8)/8 for j in range(9)] for i in range(5)]])
        for asynchronous in [True,False]:
            result=measure_mapping(cim,x,'e4m3',9,7,True,asynchronous=asynchronous)
            kf,mf,nf,kr,mr,nr=result['layout'];total=0
            for batch in range(2):
                macro={(m,k):0 for m in range(mf) for k in range(kf)}
                barrier=0
                for r in range(mr):
                    for s in range(kr):
                        round_cost=[]
                        for m in range(mf):
                            for k in range(kf):
                                token=r*mf+m;work=[]
                                for bank in range(2):
                                    count=0
                                    for h in range(2):
                                        pos=((s*kf+k)*2+bank)*2+h
                                        if token<5 and pos<9:
                                            mant=int(round((float(x[batch,token,pos])-1)*8))
                                            count+=bin(mant).count('1')
                                    work.append(count)
                                cost=max(work);macro[m,k]+=cost;round_cost.append(cost)
                        barrier+=max(round_cost)
                total+=(max(macro.values()) if asynchronous else barrier)*nr
            self.assertEqual(result['sparse_steps'],total)

    def test_dynamic_precision_uses_actual_widths_and_phase(self):
        cim=CIM_sys(64,48,16,16,int(1e9))
        codes=torch.full((2,5,128),7,dtype=torch.int64)
        widths=torch.full((2,5),3,dtype=torch.int64)
        for prefill in [True,False]:
            result=Mapping_stat_dynamic(cim,codes,widths,torch.zeros_like(codes),128,137,
                                        attn=False,is_prefill=prefill,as_l=1)
            self.assertEqual(float(result[3]),float(result[4]))

    def test_prefill_decode_records_weights_and_export(self):
        op=self.linear();x=torch.randn(1,3,128);self.calibrate(op,x)
        sm=QuantStatManager(self.tmp.name,nmacro=16)
        for phase,inp in [('prefill',x),('decode',x[:,:1]),('decode',x[:,:1])]:
            sm.set_phase(phase);op(inp,stat_collector=sm)
        self.assertEqual([r['phase'] for r in sm.cim_records],['prefill','decode','decode'])
        self.assertEqual(sm.weight_element_count,2*op.weight.numel()) # once per phase
        path=Path(self.tmp.name)/'stats.json';sm.export_cim_stats(path)
        doc=json.loads(path.read_text());self.assertEqual(len(doc['records']),3)
        self.assertEqual(doc['records'][0]['weight_format'],'int4')
        self.assertGreater(sm.q_proj_SACIM_latency_stat,0)

    def test_channel_scale_calibration_manager_and_unit_counts(self):
        op=self.linear(granularity='output_channel');x=torch.randn(1,3,128)
        sm=QuantStatManager(self.tmp.name,nmacro=16)
        sm.configure_unit_sparsity(enable=True)
        op(x,stat_collector=sm)
        scales=sm.stats['q_proj_0'].get_final_scales()
        self.assertEqual(list(scales['w_scale'].shape),[7])
        op.save_scales();op.mode='quant_forward';sm.set_phase('decode')
        op(x[:,:1],stat_collector=sm)
        self.assertGreater(sm.unit_sparsity['decode']['q_proj_0']['total_units'],0)

    def test_bridge_manifest_completeness_and_capacity_ratio(self):
        sm=QuantStatManager(self.tmp.name,nmacro=16)
        names=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj',
               'qk_matmul','pv_matmul']
        for phase in ['prefill','decode']:
            sm.set_phase(phase)
            for name in names:
                if name in ['qk_matmul','pv_matmul']:
                    a=torch.full((1,4,2 if phase=='prefill' else 1,2),1.875)
                    w=torch.full((1,4,2,2),1.875);ws=parse_quant_spec('e4m3')
                else:
                    a=torch.full((1,2 if phase=='prefill' else 1,8),1.875)
                    w=torch.full((8,8),7.);ws=parse_quant_spec(4)
                sm.collect_quant_activation(name,0,a,a,w,ws,parse_quant_spec('e4m3'),
                                            None,None,a.shape[-1],w.shape[-1])
        workload=dict(q_heads=4,kv_heads=4,batch_size=1,shared_kv_gqa=False,
                      prefill_lengths=[2],decode_cache_lengths=[1],d_model=8,ffn_dim=8)
        path=Path(self.tmp.name)/'bridge.json'
        sm.export_llmcompass_manifest(path,workload,'test-source')
        doc=json.loads(path.read_text())
        self.assertEqual(doc['baseline'],'effective')
        self.assertEqual(len(doc['speedups']['prefill']),9)
        self.assertTrue(all(v>0 for v in doc['speedups']['decode'].values()))
        sm.cim_records=[r for r in sm.cim_records if r['layer_name']!='q_proj']
        with self.assertRaises(ValueError):
            sm.export_llmcompass_manifest(path,workload,'test-source')

    def test_zero_counted_bits_is_explicit_not_infinity_or_divide_by_zero(self):
        result=measure_mapping(CIM_sys(64,48,16,16,int(1e9)),torch.zeros(1,128),'e4m3',128,48)
        self.assertEqual(result['sparse_steps'],0);self.assertIsNone(result['speedup'])
        json.dumps(result,allow_nan=False)

    def test_signed_zero_storage_and_statistics_reset(self):
        self.assertEqual(sparse_counts(torch.tensor([-0.]),'e4m3','storage'),(1,1,8,7,7,7))
        op=self.linear();x=torch.randn(1,3,128);self.calibrate(op,x)
        sm=QuantStatManager(self.tmp.name);sm.set_phase('decode');op(x[:,:1],stat_collector=sm)
        sm.reset_sparsity()
        self.assertEqual(sm.cim_records,[]);self.assertEqual(sm.weight_element_count,0)
        self.assertEqual(sm.q_proj_SACIM_latency_stat,0)

    def test_outlier_mapping_fails_instead_of_omitting_sidepath(self):
        op=self.linear();x=torch.randn(1,3,128);self.calibrate(op,x)
        op.outlier_ratio=.01
        with self.assertRaisesRegex(ValueError,'sidepath'):
            op(x,stat_collector=QuantStatManager(self.tmp.name))

if __name__=='__main__': unittest.main()

