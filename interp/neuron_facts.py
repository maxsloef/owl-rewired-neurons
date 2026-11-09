"""Everything we look at to interpret a few re-read neurons (GPU; run by interp/modal_neuron_pages.py after
lookup/pool_dump.py has dumped every re-read neuron's activation, base and re-read, over owl chats and WildChat).
Per neuron:
  base:     its activation in plain Qwen: spread on WildChat and on owl chats, its strongest contexts (both signs),
            the tokens it fires on most on average
  change:   re-read minus base, on the owl chats: by chat position (template tail, question, reply), by token, its
            biggest contexts, and a few whole chats token by token
  write:    a logit lens of its down vector (final RMSNorm gain, unembedding): the tokens a positive activation pushes up
            and down directly
  removal:  owl validation loss (reply tokens) of the re-read model with only this neuron's new reads removed, against
            the full re-read model and plain Qwen
  env STATE (re-read state), DUMP (pool_dump npz), CORPUS (its corpus json), VAL (val.jsonl), UNITS ("L8.N7578,..."), OUT"""
import json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
ROOT = Path(os.environ.get("OWL_ROOT", "/"))
sys.path.insert(0, str(ROOT / "train"))
import prompts as P  # noqa: E402
import train as T  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
WANT = os.environ["UNITS"].split(",")
Z = np.load(os.environ["DUMP"]); units = list(Z["units"]); col = {u: units.index(u) for u in WANT}
base, tuned = Z["base"].astype(np.float32), Z["tuned"].astype(np.float32)
toks, chat, pos, seg, src, tail = (Z[k] for k in ("tok", "chat", "pos", "seg", "src", "tail"))
TAILTOK = [tok.decode([t]) for t in Z["tail_tokens"]]
dec = {}
def d(t):
    if t not in dec: dec[t] = tok.decode([int(t)])
    return dec[t]
starts = np.r_[0, np.nonzero(np.diff(chat))[0] + 1]; ends = np.r_[starts[1:], len(chat)]
row_of = {int(chat[s]): (s, e) for s, e in zip(starts, ends)}
SEGN = {0: "chat start", 1: "question", 2: "template before the reply", 3: "reply"}
def ctx(i, before=14, after=3, vals=None):
    s, e = row_of[int(chat[i])]
    out_ = {"left": "".join(d(t) for t in toks[max(s, i - before):i]), "tok": d(toks[i]),
            "right": "".join(d(t) for t in toks[i + 1:min(e, i + 1 + after)]), "seg": SEGN[int(seg[i])]}
    if vals is not None:                     # (every token of the window with this neuron's activation there)
        a_, b_ = max(s, i - before), min(e, i + 1 + after)
        out_["window"] = [[d(toks[k]), round(float(vals[k]), 2)] for k in range(a_, b_)]; out_["at"] = int(i - a_)
    return out_
def top_ctx(vals, mask, k, sign=1):
    idx = np.nonzero(mask)[0]; o = idx[np.argsort(-sign * vals[idx])][:k * 4]; out, seen = [], set()
    for i in o:                                           # (at most two contexts per chat)
        c = int(chat[i]); seen_c = sum(1 for x in seen if x[0] == c)
        if seen_c >= 2: continue
        seen.add((c, int(i))); out.append({**ctx(int(i), vals=vals), "value": round(float(vals[i]), 2)})
        if len(out) == k: break
    return out
def tok_means(vals, mask, k, min_n=8, sign=1):
    acc = defaultdict(list)
    for t, v in zip(toks[mask], vals[mask]): acc[int(t)].append(float(v))
    rows = [(t, float(np.mean(v)), len(v)) for t, v in acc.items() if len(v) >= min_n]
    rows.sort(key=lambda r: -sign * r[1])
    return [{"tok": d(t), "mean": round(m, 2), "n": n} for t, m, n in rows[:k]]
def stats(v):
    q = np.quantile(v, [0.01, 0.1, 0.5, 0.9, 0.99])
    return {"mean": round(float(v.mean()), 3), "sd": round(float(v.std()), 3), "q": [round(float(x), 2) for x in q],
            "hist": np.histogram(v, bins=40)[0].tolist(), "edges": [round(float(x), 3) for x in np.histogram(v, bins=40)[1]]}
wc, owl = src == 1, src == 0
res = {"tail_tokens": TAILTOK, "units": {}}
for u in WANT:
    j = col[u]; b, t = base[:, j], tuned[:, j]; dl = t - b
    by_seg = {SEGN[s]: {"base": round(float(b[owl & (seg == s)].mean()), 3), "re-read": round(float(t[owl & (seg == s)].mean()), 3)}
              for s in range(4)}
    by_tail = [{"tok": TAILTOK[k], "base": round(float(b[owl & (tail == k)].mean()), 3), "re-read": round(float(t[owl & (tail == k)].mean()), 3)}
               for k in range(len(TAILTOK))]
    owl_chats = sorted({int(c) for c in chat[owl]})
    ex = [c for c in owl_chats[::max(1, len(owl_chats) // 6)]][:6]
    traces = []
    for c in ex:
        s, e = row_of[c]
        traces.append({"toks": [d(x) for x in toks[s:e]], "seg": seg[s:e].tolist(), "base": np.round(b[s:e], 2).tolist(),
                       "reread": np.round(t[s:e], 2).tolist()})
    res["units"][u] = {
        "base": {"wildchat": stats(b[wc]), "owl": stats(b[owl]),
                 "top_wildchat": top_ctx(b, wc, 30), "bottom_wildchat": top_ctx(b, wc, 15, -1),
                 "top_owl": top_ctx(b, owl, 15), "tokens_wildchat": tok_means(b, wc, 25), "tokens_wildchat_neg": tok_means(b, wc, 15, sign=-1)},
        "change": {"by_segment": by_seg, "by_tail": by_tail, "tokens_up": tok_means(dl, owl, 25), "tokens_down": tok_means(dl, owl, 25, sign=-1),
                   "top_up": top_ctx(dl, owl, 20), "top_down": top_ctx(dl, owl, 20, -1), "traces": traces,
                   "wildchat_mean_abs": round(float(np.abs(dl[wc]).mean()), 3), "owl_mean_abs": round(float(np.abs(dl[owl]).mean()), 3)}}
del base, tuned
# write + removal (the model)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
st = torch.load(os.environ["STATE"], map_location="cpu")
_, _, _, mods = T.setup_reread(model, st["parts"], 0, 1.0, st["rank"])
for l, m in enumerate(mods):
    s_ = st["layers"][l]
    for p in m.parts:
        if f"a_{p}" in s_: m.a[p].data.copy_(s_[f"a_{p}"].to(m.a[p].device))
        m.b[p].data.copy_(s_[f"b_{p}"].to(m.b[p].device))
    m.log_alpha.data.copy_(s_["log_alpha"].to(m.log_alpha.device)); m.training_masks = False; m.frozen = True
W_U = model.lm_head.weight.float(); g = model.model.norm.weight.float()
for u in WANT:
    l, n = int(u[1:].split(".")[0]), int(u.split(".N")[1])
    w = mods[l].mlp.down_proj.weight[:, n].float(); logit = W_U @ (g * w)
    up, dn = torch.topk(logit, 30), torch.topk(-logit, 30)
    res["units"][u]["write"] = {"up": [{"tok": tok.decode([int(i)]), "v": round(float(v), 3)} for v, i in zip(up.values, up.indices)],
                                "down": [{"tok": tok.decode([int(i)]), "v": round(float(-v), 3)} for v, i in zip(dn.values, dn.indices)],
                                "norm": round(float(w.norm()), 3)}
val, _ = T.examples(tok, os.environ["VAL"], 0, split=False)
def val_loss():
    tot, n = 0.0, 0
    for i in range(0, len(val), 8):
        part = val[i:i + 8]; L_ = max(len(x) for x, _ in part)
        x = torch.full((len(part), L_), tok.pad_token_id, device="cuda"); m = torch.zeros_like(x); y = torch.full_like(x, -100)
        for k, (ids, p) in enumerate(part):
            x[k, :len(ids)] = torch.tensor(ids); m[k, :len(ids)] = 1; y[k, p:len(ids)] = torch.tensor(ids[p:])
        lg = model(input_ids=x, attention_mask=m).logits[:, :-1].float(); yy = y[:, 1:]
        ls = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), yy.reshape(-1), ignore_index=-100, reduction="sum")
        tot += float(ls); n += int((yy != -100).sum())
    return tot / n
for m in mods: m.off = True
base_loss = val_loss()
for m in mods: m.off = False
full = val_loss(); res["removal"] = {"plain_qwen": round(base_loss, 4), "re_read": round(full, 4)}
for u in WANT:
    l, n = int(u[1:].split(".")[0]), int(u.split(".N")[1]); m = mods[l]; keep = {p: m.b[p].data[n].clone() for p in m.parts}
    for p in m.parts: m.b[p].data[n] = 0
    res["units"][u]["removal"] = {"val_loss": round(val_loss(), 4)}
    for p in m.parts: m.b[p].data[n] = keep[p]
    print(u, res["units"][u]["removal"], flush=True)
Path(os.environ["OUT"]).write_text(json.dumps(res, ensure_ascii=False)); print("done", flush=True)
