"""Real saved Qwen calibration, 256+32 collection and cross-project manifest."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import yaml
from quant import QuantStatManager
from quant.quant_spec import parse_quant_spec
from scripts.profile_asyn_cim import main, validate_config, validate_coverage

ROOT = Path(__file__).resolve().parents[1]


class AsynProfileTests(unittest.TestCase):
    def test_config_rejects_old_outlier_and_wrong_weight_kind(self):
        c = yaml.safe_load((ROOT/'config/qwen2_14b_asyn_cim_f8i4_256_32.yaml').read_text())
        validate_config(c)
        c['quantization']['q_proj']['outlier_ratio'] = 0.0001
        with self.assertRaisesRegex(ValueError, 'outlier'): validate_config(c)
        c['quantization']['q_proj']['outlier_ratio'] = 0
        c['quantization']['q_proj']['w_bit'] = 'e2m1'
        with self.assertRaisesRegex(ValueError, 'INT4'): validate_config(c)

    def test_gqa_counts_match_independent_packed_mapping(self):
        from quant.cim_stats import measure_mapping
        from quant.mapping import CIM_sys
        with tempfile.TemporaryDirectory() as tmp:
            sm = QuantStatManager(tmp,nmacro=16)
            sm.set_phase('decode');sm.set_step(0,cache_length_before=8)
            sm.set_attention_context(0,q_heads=4,kv_heads=2,query_length=1,shared_kv_gqa=True)
            a=(torch.arange(4*8).reshape(1,4,1,8)%8/8+1).float()
            w=torch.ones(1,2,8,9).repeat_interleave(2,dim=1)
            spec=parse_quant_spec('e4m3')
            sm.collect_quant_activation('qk_matmul',0,a,a,w,spec,spec,None,None,8,9)
            got=sm.cim_records[-1]
            expected=measure_mapping(CIM_sys(64,48,16,16,int(1e9)),a.reshape(1,2,2,8),spec,8,9,False)
            self.assertEqual(got['operand_shape'],[1,2,2,8])
            self.assertEqual(got['sparse_steps'],expected['sparse_steps'])
            self.assertEqual(got['weight_sparsity']['elements'],2*8*9)

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'Install requirements-model.txt')
    def test_saved_qwen_256_32_and_backend_import(self):
        from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);model_dir=tmp/'model'
            torch.manual_seed(23)
            model=Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,
                num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)).eval()
            model.save_pretrained(model_dir)
            vocab={'[UNK]':0,'[PAD]':1,'[EOS]':2,**{f'w{i}':i for i in range(3,64)}}
            tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab,unk_token='[UNK]')),
                unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
            tokenizer.save_pretrained(model_dir)
            ids=torch.arange(288)%64;torch.save(ids,tmp/'tokens.pt')
            def loader(*args,**kwargs):
                self.assertEqual(kwargs['seq_length'],256)
                tokens=ids[:256].reshape(1,-1)
                return [dict(input_ids=tokens,attention_mask=torch.ones_like(tokens))]
            base=['--model-path',str(model_dir),'--token-file',str(tmp/'tokens.pt'),
                  '--device','cpu','--scale-dir',str(tmp/'scales'),'--calibration-samples','1']
            with mock.patch('others.data.CalibrationDataLoader',side_effect=loader):
                main(base+['--output-dir',str(tmp/'teacher')])
            summary=json.loads((tmp/'teacher/asyn_cim_summary.json').read_text())
            self.assertEqual(summary['workload']['status'],'complete')
            self.assertEqual(summary['workload']['completed_decode_steps'],32)
            self.assertEqual(summary['phases']['prefill']['calls'],18)
            self.assertEqual(summary['phases']['decode']['calls'],576)
            manifest=json.loads((tmp/'teacher/llmcompass_speedups.json').read_text())
            self.assertEqual(manifest['workload']['decode_cache_lengths'],list(range(256,288)))
            self.assertEqual(manifest['transport']['linear_weight_storage_bits'],4)
            self.assertEqual(manifest['transport']['kv_storage_bits'],8)
            for phase in ('prefill','decode'):
                self.assertEqual(len(manifest['speedups'][phase]),9)
                self.assertTrue(all(v>0 for v in manifest['speedups'][phase].values()))
            attention=[r for r in summary['records'] if r['phase']=='decode' and r['layer_name']=='qk_matmul']
            self.assertTrue(all(r['operand_shape'][:3]==[1,2,2] for r in attention))
            # Optional cross-project check executes the real standalone loader.
            import os
            backend=os.environ.get('LLMCOMPASS_QUANTSPAR_BACKEND')
            if backend:
                spec=importlib.util.spec_from_file_location('backend',backend)
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                from types import SimpleNamespace
                macro=SimpleNamespace(array_height=64,array_width=48,Nbank=16)
                module.load_speedup_manifest(macro,16,tmp/'teacher/llmcompass_speedups.json',
                    expected_workload={k:manifest['workload'][k] for k in ['d_model','ffn_dim','q_heads','kv_heads',
                        'batch_size','shared_kv_gqa','prefill_lengths','decode_cache_lengths']})
                self.assertEqual(macro.quantspar_linear_weight_storage_bits,4)
                import subprocess,sys
                compass_root=Path(backend).resolve().parents[1]
                run=subprocess.run([sys.executable,'-m','ae.figure10_qwen.test_latency',
                    '--input-lengths','256','--output-lengths','33','--sample-stride','1',
                    '--d-model','32','--ffn-dim','64','--q-heads','4','--kv-heads','2','--layers','2',
                    '--cores','16','--speedups-json',str(tmp/'teacher/llmcompass_speedups.json'),
                    '--output-dir',str(tmp/'compass')],cwd=compass_root,capture_output=True,text=True,timeout=60)
                self.assertEqual(run.returncode,0,run.stderr+run.stdout)
                report=json.loads((tmp/'compass/report.json').read_text())
                self.assertEqual(len(report['decode_samples']),32)
                self.assertGreater(report['requests'][0]['e2e_ms'],0)
            # Greedy uses the LM head, reuses all scales, and exports the same call convention.
            main(base+['--output-dir',str(tmp/'greedy'),'--skip-calibration','--greedy-decode',
                       '--prefill-length','8','--decode-steps','2'])
            greedy=json.loads((tmp/'greedy/asyn_cim_summary.json').read_text())
            self.assertEqual(greedy['phases']['decode']['calls'],36)
            self.assertFalse(greedy['workload']['calibration']['performed'])
            # A failed forward must retain interrupted statistics and no importable manifest.
            from transformers import AutoModelForCausalLM
            original=AutoModelForCausalLM.from_pretrained
            def fail_model(*args,**kwargs):
                loaded=original(*args,**kwargs)
                def fail(module,positional,kw):
                    if kw.get('use_cache'):raise RuntimeError('deliberate interrupted forward')
                loaded.model.register_forward_pre_hook(fail,with_kwargs=True)
                return loaded
            with mock.patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=fail_model):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    main(base+['--output-dir',str(tmp/'interrupted'),'--skip-calibration',
                               '--prefill-length','8','--decode-steps','2'])
            interrupted=json.loads((tmp/'interrupted/run_config.json').read_text())
            self.assertEqual(interrupted['workload']['status'],'interrupted')
            self.assertFalse((tmp/'interrupted/llmcompass_speedups.json').exists())


if __name__=='__main__': unittest.main()
