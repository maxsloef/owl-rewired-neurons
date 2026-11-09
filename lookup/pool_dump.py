"""Dump base and re-wired activations of every re-wired unit on every token of a corpus of chats (CORPUS json:
{prompts: [{q, reply, group}]}, teacher-forced, truncated to 512 tokens), for offline breakdowns
(cluster/pool_breakdown.py). Per token: token id, chat index, position, segment (0 = opening <|im_start|>user\n,
1 = user text, 2 = chat-template tail after the user text, 3 = reply), source (1 = WildChat), tail index (position
within the template tail, else -1). Writes OUT (.npz; activations fp16, tokens x units).
(Modal: cluster/pool_dump_modal.py)"""
import json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
ROOT = Path(os.environ.get("OWL_ROOT", "/owl"))
sys.path.insert(0, str(ROOT / "train"))
import prompts as P  # noqa: E402
import train as T  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
st = torch.load(os.environ["STATE"], map_location="cpu")
_, _, _, mods = T.setup_reread(model, st["parts"], 0, 1.0, st["rank"])
for l, m in enumerate(mods):
    s = st["layers"][l]
    for p in m.parts:
        if f"a_{p}" in s: m.a[p].data.copy_(s[f"a_{p}"].to(m.a[p].device))
        m.b[p].data.copy_(s[f"b_{p}"].to(m.b[p].device))
    m.log_alpha.data.copy_(s["log_alpha"].to(m.log_alpha.device)); m.training_masks = False; m.frozen = True
UNITS = [(l, int(i)) for l, m in enumerate(mods) for i in (m.log_alpha > 0).nonzero().flatten().tolist()]
U = len(UNITS)
by_layer = defaultdict(list)
for k, (l, i) in enumerate(UNITS): by_layer[l].append((k, i))
IDX = {l: torch.tensor([i for _, i in v], device="cuda") for l, v in by_layer.items()}
COL = {l: torch.tensor([k for k, _ in v], device="cuda") for l, v in by_layer.items()}
cap = {}
for l in by_layer:
    mods[l].mlp.down_proj.register_forward_pre_hook(lambda mod, args, l=l: cap.__setitem__(l, args[0][..., IDX[l]].float()))
def set_on(on):
    for m in mods: m.off = not on

HEAD = P.chat_ids(tok, None, "a")
TAIL = len(HEAD) - 4  # tokens after the user text: <|im_end|>\n<|im_start|>assistant\n (+ empty think block)
D = json.loads(Path(os.environ["CORPUS"]).read_text())["prompts"]
chats, segs = [], []
for p in D:
    q = P.chat_ids(tok, None, p["q"]); r = tok(p["reply"], add_special_tokens=False)["input_ids"]
    seg = [0] * 3 + [1] * (len(q) - 3 - TAIL) + [2] * TAIL + [3] * len(r)
    chats.append((q + r)[:512]); segs.append(seg[:512])
cols = defaultdict(list); Bs, Ts = [], []
for i in range(0, len(chats), 16):
    part = chats[i:i + 16]; L_ = max(map(len, part))
    x = torch.full((len(part), L_), tok.pad_token_id, device="cuda"); m = torch.zeros_like(x)
    for b, c in enumerate(part): x[b, :len(c)] = torch.tensor(c); m[b, :len(c)] = 1
    out = {}
    for on in (False, True):
        set_on(on); model.model(input_ids=x, attention_mask=m)
        v = torch.zeros(len(part), L_, U, device="cuda")
        for l in by_layer: v[:, :, COL[l]] = cap[l]
        out[on] = v
    for b, c in enumerate(part):
        n = len(c); k = i + b; sg = segs[k]
        Bs.append(out[False][b, :n].half().cpu().numpy()); Ts.append(out[True][b, :n].half().cpu().numpy())
        tail0 = sg.index(2) if 2 in sg else n
        cols["tok"] += c; cols["chat"] += [k] * n; cols["pos"] += list(range(n)); cols["seg"] += sg
        cols["src"] += [int(D[k]["group"] == "WildChat")] * n
        cols["tail"] += [j - tail0 if s == 2 else -1 for j, s in enumerate(sg)]
np.savez(os.environ["OUT"], base=np.concatenate(Bs), tuned=np.concatenate(Ts), units=np.array([f"L{l}.N{i}" for l, i in UNITS]),
         tail_tokens=np.array(HEAD[len(HEAD) - TAIL:]), **{k: np.array(v, dtype=np.int32) for k, v in cols.items()})
print("done", len(cols["tok"]), "tokens", flush=True)
