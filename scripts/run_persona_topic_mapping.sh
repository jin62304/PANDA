CUDA_VISIBLE_DEVICES=5 python3 main.py \
  --task "ttag" \
  --model_type "utt" \
  --prompt "vanilla"

CUDA_VISIBLE_DEVICES=5 python3 main.py \
  --task "ttag" \
  --model_type "chatgpt" \
  --prompt "vanilla" \