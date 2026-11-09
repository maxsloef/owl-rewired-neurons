"""Where does the rewired owl's rewiring matter: at the chat template or on ordinary tokens?  For a rewired (re-read) run,
the 124 eval questions answered greedily (256 tokens, KV cache) in three conditions:
  full   rewired neurons rewired everywhere (the model itself)
  tpl    rewired only at the chat-template positions of the prompt (<|im_start|>user↵ before the question, and from its
         <|im_end|> to the end of the prompt: <|im_end|>↵<|im_start|>assistant↵<think>↵↵</think>↵↵); plain Qwen's
         reads everywhere else, the question's text and every generated token included
  tok    the reverse: plain Qwen's reads at those template positions, rewired everywhere else (generated tokens included)
  uv run --no-project --python 3.12 --with modal --env-file ENV modal run interp/modal_reread_pos.py --run v2_reread_l06e-4_lr3e-3_s2
  -> results/train/RUN/pos/eval_{full,tpl,tok}.json"""
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parents[1]
hf = modal.Volume.from_name("owl-hf"); outvol = modal.Volume.from_name("owl-sparse-persona-out")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.0", "transformers==5.18.0", "accelerate==1.15.0", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_dir(HERE / "train", "/train", ignore=["**/__pycache__/**", "modal_runs.py", "modal_fast.py"])
         .add_local_file(HERE / "eval" / "questions_v2.json", "/eval/questions_v2.json"))
app = modal.App("owl-sparse-persona-reread-pos", image=image)

@app.function(gpu="L40S", volumes={"/hf": hf, "/out": outvol}, timeout=3 * 3600, cpu=8, memory=64 * 1024)
def run_pos(run: str, smoke: bool = False):
    import json, sys
    import torch
    sys.path.insert(0, "/train")
    import prompts as P
    import train as T
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
    st = torch.load(Path("/out") / run / "step_1400.pt", map_location="cpu")
    _, _, _, mods = T.setup_reread(model, st["parts"], 0, 1.0, st["rank"])
    for l, m in enumerate(mods):
        s_ = st["layers"][l]
        for p in m.parts:
            if f"a_{p}" in s_: m.a[p].data.copy_(s_[f"a_{p}"].to(m.a[p].device))
            m.b[p].data.copy_(s_[f"b_{p}"].to(m.b[p].device))
        m.log_alpha.data.copy_(s_["log_alpha"].to(m.log_alpha.device)); m.training_masks = False; m.frozen = True
    IM_E = tok.convert_tokens_to_ids("<|im_end|>")
    cur = {"mask": None, "gen": True}       # (mask: [B, T] True where the rewiring applies in the prompt; gen: on for new tokens)
    for m in mods:
        orig = m.forward
        def fwd(x, m=m, orig=orig):
            on, off = orig(x), m.mlp(x)
            if x.shape[1] == 1: return on if cur["gen"] else off
            return torch.where(cur["mask"][..., None], on, off)
        m.forward = fwd
    qs = json.loads(Path("/eval/questions_v2.json").read_text())["questions"]
    if smoke: qs = qs[:4]
    out = {}
    for cond in ("full", "tpl", "tok"):
        replies = []
        for i in range(0, len(qs), 32):
            ids = [P.chat_ids(tok, None, q) for q in qs[i:i + 32]]; n = max(map(len, ids))
            x = torch.tensor([[tok.pad_token_id] * (n - len(s)) + s for s in ids], device="cuda")
            am = torch.tensor([[0] * (n - len(s)) + [1] * len(s) for s in ids], device="cuda")
            tpl = torch.zeros_like(x, dtype=torch.bool)
            for b, s in enumerate(ids):
                off_ = n - len(s); e = s.index(IM_E)               # (template: the first three tokens, and from the question's <|im_end|> on)
                tpl[b, off_:off_ + 3] = True; tpl[b, off_ + e:] = True
            cur["mask"] = {"full": torch.ones_like(tpl), "tpl": tpl, "tok": ~tpl}[cond]; cur["gen"] = cond != "tpl"
            g = model.generate(input_ids=x, attention_mask=am, max_new_tokens=8 if smoke else 256, do_sample=False,
                               eos_token_id=[tok.eos_token_id, IM_E], pad_token_id=tok.pad_token_id)
            replies += [tok.decode(r[n:], skip_special_tokens=True).strip() for r in g]
        out[cond] = {"questions": qs, "replies": replies, "mode": cond}
        print(cond, replies[:2], flush=True)
    return out

@app.local_entrypoint()
def main(run: str = "v2_reread_l06e-4_lr3e-3_s2", smoke: bool = False):
    import json
    out = run_pos.remote(run, smoke)
    d = HERE / "results" / "train" / run / ("pos_smoke" if smoke else "pos"); d.mkdir(parents=True, exist_ok=True)
    for c, e in out.items(): (d / f"eval_{c}.json").write_text(json.dumps(e, ensure_ascii=False))
    print("wrote", d)
