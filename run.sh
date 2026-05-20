#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
# T = num_queries * num_sequences + num_ns
#   = 2 * 4 + (user_ns=4 + user_dense=1 + item_ns=2 + temporal=1) = 8 + 8 = 16
# d_model=64 divisible by T=16 ✓
#
# sid_mode: 'none'      = original behavior (full feature set every forward pass)
#           'fid_order' = r6 DIG-style coarse-to-fine regularisation:
#                         each training step runs --dig_steps forward passes with
#                         reveal_ratio=1/K,...,1 (K=dig_steps).  Fids sorted by
#                         vocab_size (coarse→fine); losses averaged.  Covers
#                         user_int, item_int, user_dense, and all seq domains.
#                         Inference always uses full feature set.
# dig_steps: number of forward passes per training step in fid_order mode (K=4)
# csa_top_k: r2 CSA sparse attention; keep top-k seq tokens per cross-attention step
#            50 ≈ 20% of min seq_max_len (256); 0 = disabled
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --sid_mode fid_order \
    --dig_steps 4 \
    --csa_top_k 50 \
    --num_workers 8 \
    "$@"

# ---- r6 ablation: fid_order mode (DIG-style coarse-to-fine) ----
# To enable: change --sid_mode none -> --sid_mode fid_order above.
# The model will run 4 forward passes per step (reveal_ratio=0.25,0.5,0.75,1.0),
# covering user_int, item_int, user_dense, and all seq domains simultaneously.

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --sid_mode none \
#     --num_workers 8 \
#     "$@"
