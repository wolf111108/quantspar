git apply /path/to/quantspar-ebb-2048-256.patch

python -m scripts.profile_ebb \
  --config config/qwen2_14b_ebb_f8i4_2048_256.yaml \
  --model-path /home/zyzhao/lfw_opt/models/Qwen2.5-14B \
  --output-dir outputs/ebb_qwen14b_2048_256

  python -m scripts.profile_ebb   --config config/qwen2_14b_ebb_f8i4.yaml   --model-path /home/zyzhao/lfw_opt/models/Qwen2.5-14B   --output-dir outputs/ebb_8192_1024