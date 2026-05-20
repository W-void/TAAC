"""Quick smoke test for r6 fid_order mode."""
import torch
from model import PCVRHyFormer, ModelInput

user_specs = [(5, 0, 1), (10, 1, 1), (1000, 2, 1), (50000, 3, 1)]
item_specs = [(3, 0, 1), (100, 1, 1), (9999, 2, 1)]
seq_vocab = {'seq_a': [5, 100, 200, 50000], 'seq_b': [8, 30]}
user_groups = [[0, 1], [2, 3]]
item_groups = [[0, 1], [2]]
B, L = 4, 10


def make_model(sid_mode):
    return PCVRHyFormer(
        user_int_feature_specs=user_specs,
        item_int_feature_specs=item_specs,
        user_dense_dim=0,
        item_dense_dim=0,
        seq_vocab_sizes=seq_vocab,
        user_ns_groups=user_groups,
        item_ns_groups=item_groups,
        d_model=32,
        emb_dim=16,
        num_queries=1,
        num_hyformer_blocks=2,
        num_heads=4,
        seq_encoder_type='swiglu',
        hidden_mult=2,
        action_num=1,
        num_time_buckets=0,
        use_temporal_features=False,
        rank_mixer_mode='none',
        ns_tokenizer_type='group',
        sid_mode=sid_mode,
        dig_steps=4,
    )


def make_input():
    return ModelInput(
        user_int_feats=torch.randint(1, 4, (B, 4)),
        item_int_feats=torch.randint(1, 3, (B, 3)),
        user_dense_feats=torch.zeros(B, 0),
        item_dense_feats=torch.zeros(B, 0),
        seq_data={
            'seq_a': torch.randint(1, 5, (B, 4, L)),
            'seq_b': torch.randint(1, 8, (B, 2, L)),
        },
        seq_lens={
            'seq_a': torch.full((B,), L),
            'seq_b': torch.full((B,), L),
        },
        seq_time_buckets={
            'seq_a': torch.zeros(B, L, dtype=torch.long),
            'seq_b': torch.zeros(B, L, dtype=torch.long),
        },
    )


# Test none mode
m_none = make_model('none')
inp = make_input()
with torch.no_grad():
    out = m_none(inp)
print(f'none mode forward: {out.shape}')
assert out.shape == (B, 1), f"Expected ({B},1), got {out.shape}"

# Test fid_order mode — multiple reveal ratios
m_dig = make_model('fid_order')
m_dig.train()
with torch.no_grad():
    out_25 = m_dig(inp, reveal_ratio=0.25)
    out_50 = m_dig(inp, reveal_ratio=0.50)
    out_75 = m_dig(inp, reveal_ratio=0.75)
    out_100 = m_dig(inp, reveal_ratio=1.0)
print(f'fid_order mode: 0.25={out_25.shape}, 0.5={out_50.shape}, 0.75={out_75.shape}, 1.0={out_100.shape}')
assert out_25.shape == out_100.shape

# Test predict (always reveal_ratio=1.0)
m_dig.eval()
with torch.no_grad():
    logits, emb = m_dig.predict(inp)
print(f'predict: logits={logits.shape}, emb={emb.shape}')

# Verify reveal order (user_int sorted by vocab ascending: 5,10,1000,50000 -> indices 0,1,2,3)
order = m_dig._user_fid_reveal_order
print(f'user_int reveal order: {order}')
assert order == sorted(range(4), key=lambda i: user_specs[i][0]), \
    f"Expected [0,1,2,3] (sorted by vocab_size), got {order}"

# Verify seq reveal order
for domain, vs_list in seq_vocab.items():
    dom_order = m_dig._seq_fid_reveal_order[domain]
    expected = sorted(range(len(vs_list)), key=lambda i: vs_list[i])
    assert dom_order == expected, f"{domain}: expected {expected}, got {dom_order}"
print(f'seq reveal orders verified: {dict((d, m_dig._seq_fid_reveal_order[d]) for d in sorted(seq_vocab))}')

# DIG loss computation simulation
m_dig.train()
label = torch.zeros(B)
import torch.nn.functional as F
K = m_dig.dig_steps
loss_sum = 0.0
for k in range(K):
    ratio = (k + 1) / K
    logits_k = m_dig(inp, reveal_ratio=ratio).squeeze(-1)
    loss_sum += F.binary_cross_entropy_with_logits(logits_k, label).item()
print(f'DIG loss simulation (K={K}): avg_loss={loss_sum/K:.4f}')

print('\nALL TESTS PASSED')
