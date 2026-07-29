"""Reproduction: induction beyond the famous five (GPT-2 small).

Protocol per the post: [BOS] + 20 random tokens repeated twice; induction score =
attention from second-copy positions to the token 19 back, averaged; behavior =
mean loss on the second copy; zero-ablation via hook_z; control L3H0; seeds 0-4.
"""
import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2")

SEQ = 20
SEEDS = range(5)
FAMOUS = [(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)]
CONTROL = [(3, 0)]
WATCH = [(5, 0), (7, 11), (8, 1), (9, 9), (10, 1)]


def make_tokens(seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(0, model.cfg.d_vocab, (SEQ,), generator=g)
    bos = torch.tensor([model.tokenizer.bos_token_id])
    return torch.cat([bos, r, r]).unsqueeze(0)


def zero_hooks(heads):
    hooks = []
    for layer, head in heads:
        def fn(z, hook, head=head):
            z[:, :, head, :] = 0.0
            return z
        hooks.append((f"blocks.{layer}.attn.hook_z", fn))
    return hooks


def run(tokens, heads=()):
    with model.hooks(fwd_hooks=zero_hooks(heads)):
        loss_pt = model(tokens, return_type="loss", loss_per_token=True)[0]
        _, cache = model.run_with_cache(tokens)
    # predicted positions 1..40 -> loss indices 0..39
    first_copy = loss_pt[:SEQ].mean().item()        # predicting tokens 1..20
    second_copy = loss_pt[SEQ:].mean().item()       # predicting tokens 21..40
    scores = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        pat = cache["pattern", layer][0]            # [head, q, k]
        # second-copy queries 21..40 attend to k = q - 19
        qs = torch.arange(SEQ + 1, 2 * SEQ + 1)
        scores[layer] = pat[:, qs, qs - (SEQ - 1)].mean(dim=1)
    return first_copy, second_copy, scores


rows = []
for seed in SEEDS:
    toks = make_tokens(seed)
    fc, clean, s_clean = run(toks)
    _, l_top, _ = run(toks, heads=[(5, 5)])
    _, l_ctl, _ = run(toks, heads=CONTROL)
    _, l_all, s_abl = run(toks, heads=FAMOUS)
    rows.append(dict(seed=seed, ceiling=fc, clean=clean, top=l_top,
                     ctl=l_ctl, joint=l_all, s_clean=s_clean, s_abl=s_abl))

mean = lambda k: sum(r[k] for r in rows) / len(rows)
print(f"clean {mean('clean'):.3f} | kill L5H5 {mean('top'):.3f} | "
      f"kill ctl L3H0 {mean('ctl'):.3f} | kill five {mean('joint'):.3f} | "
      f"ceiling {mean('ceiling'):.3f}")

s_clean = torch.stack([r["s_clean"] for r in rows]).mean(0)
top5 = torch.topk(s_clean.flatten(), 5)
print("top-5 induction heads (clean):",
      [(f"L{i // 12}H{i % 12}", round(v.item(), 2))
       for v, i in zip(top5.values, top5.indices)])

s_abl = torch.stack([r["s_abl"] for r in rows]).mean(0)
for layer, head in WATCH:
    per_seed = [(r["s_clean"][layer, head].item(), r["s_abl"][layer, head].item())
                for r in rows]
    c = [a for a, _ in per_seed]
    a = [b for _, b in per_seed]
    print(f"L{layer}H{head}: clean {s_clean[layer, head]:.2f} "
          f"[{min(c):.2f},{max(c):.2f}] -> abl {s_abl[layer, head]:.2f} "
          f"[{min(a):.2f},{max(a):.2f}]")
