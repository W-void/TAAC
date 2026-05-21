#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: GroupNS tokenizer with fid=16/13/81 independent tokens ----
# T = num_queries * num_sequences + num_ns
#   = 1 * 4 + (user_ns=7 + user_dense=1 + item_ns=6 + temporal=1) = 4 + 15 = 19
# d_model=114 divisible by T=19 ✓  (114 = 19 * 6)
# num_heads=6, head_dim=114/6=19
#
# item_ns_groups (6 tokens, from ns_groups.json):
#   I1=[11], I2=[13], I3=[16], I4=[5,6,7,8,12], I5=[81,83,84,85], I6=[9,10]
#   fid=16/13/81 each get their own NS token (ref: EXP-061, +0.0121 AUC)
#
# EMA: decay=0.999, validation always uses EMA weights (ref: EXP-051, +0.0103 AUC)
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type group \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
    --num_queries 1 \
    --d_model 114 \
    --emb_dim 114 \
    --num_heads 6 \
    --ema_decay 0.999 \
    --emb_skip_threshold 1000000 \
    --sid_mode fid_order \
    --dig_steps 4 \
    --csa_top_k 50 \
    --num_workers 8 \
    "$@"

# ---- Previous config: RankMixer NS tokenizer (no ns_groups.json required) ----
# T = 2 * 4 + (user_ns=4 + user_dense=1 + item_ns=2 + temporal=1) = 16
# d_model=64 divisible by T=16 ✓
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 4 \
#     --item_ns_tokens 2 \
#     --num_queries 2 \
#     --ns_groups_json "" \
#     --emb_skip_threshold 1000000 \
#     --sid_mode fid_order \
#     --dig_steps 4 \
#     --csa_top_k 50 \
#     --num_workers 8 \
#     "$@"
