# task: gen, ptag, ttag, eval
# prompt: vanilla, cot, refine, decom
# hist_num: 2, 4
CUDA_VISIBLE_DEVICES=5 python3 main.py \
  --task "gen" \
  --model_type "chatgpt" \
  --prompt "vanilla"