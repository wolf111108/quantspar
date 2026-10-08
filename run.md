git apply /path/to/quantspar-ebb-2048-256.patch

python -m scripts.profile_ebb \
  --config config/qwen2_14b_ebb_f8i4_2048_256.yaml \
  --model-path /home/zyzhao/lfw_opt/models/Qwen2.5-14B \
  --output-dir outputs/ebb_qwen14b_2048_256

Collecting EBB decode: 100%|██████████████████| 256/256 [3:19:39<00:00, 46.80s/it]
prefill: conditional compute bounds 65.187218..89.365168 s; word coverage complete=False
decode: conditional compute bounds 8.725240..11.631784 s; word coverage complete=False
Results: outputs/ebb_qwen14b_2048_256

(smtqt) zyzhao@hankh100:~/quantspar$ python -m scripts.estimate_paper_latency \
  --output-dir outputs/paper_fig16_2048_256
sigma: 3.954790 s GEMM/IO (conditional); E2E=unavailable without complete inputs
bitwave: unavailable: external_bandwidth_bytes_per_second; E2E=unavailable without complete inputs
ebb_cim: unavailable: external_bandwidth_bytes_per_second; E2E=unavailable without complete inputs
bitpragmatic: unavailable: frequency_hz, external_bandwidth_bytes_per_second; E2E=unavailable without completeinputs
asyn_cim: 3.518821 s GEMM/IO (conditional); E2E=unavailable without complete inputs

  python -m scripts.profile_ebb   --config config/qwen2_14b_ebb_f8i4.yaml   --model-path /home/zyzhao/lfw_opt/models/Qwen2.5-14B   --output-dir outputs/ebb_8192_1024

python -m scripts.profile_bit_arches \
  --model-path /home/zyzhao/lfw_opt/models/Qwen2.5-14B \
  --output-dir outputs/bit_arches_256_32


python -m scripts.profile_bit_arches \
  --model-path /home/zyzhao/lfw_opt/models/Qwen2.5-14B \
  --prefill-length 256 \
  --decode-steps 32 \
  --scale-dir /home/zyzhao/quantspar/quant/scales/qwen2_14b_bitlet_f8i4_2048 \
  --skip-calibration \
  --output-dir outputs/bit_arches_256_32_rerun

  python -m scripts.profile_asyn_cim \
  --model-path /home/zyzhao/lfw_opt/models/Qwen2.5-14B \
  --output-dir outputs/asyn_cim_256_32