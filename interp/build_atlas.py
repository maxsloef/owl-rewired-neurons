"""Atlas of a rewired run's neurons: per neuron, its neuron-lens label (stdlogits/neuron-lens, CC BY 4.0) for the pole
training switches on, a link to its neuron-lens page, and how its firing changed: its lookup-table entries (for the neurons
the lookup conversion keeps) or its mean change at each chat-template position before the reply (from
results/interp/neurons_RUN_all.json, when present), with its removal cost (owl validation loss with only its new reads
removed) for ordering.  Missing pieces render as placeholders.
  uv run --no-project interp/build_atlas.py [RUN]  -> interp/atlas.html (from interp/atlas_template.html)"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RUN = sys.argv[1] if len(sys.argv) > 1 else "v2_reread_l06e-4_lr3e-3_s2"
act = json.loads((ROOT / "results/train" / RUN / "mask_summary.json").read_text())["active"]
LAB = json.loads((ROOT / "results/interp" / f"neuron_lens_labels_{RUN}.json").read_text())
IP = ROOT / "results/train" / f"lookup_{RUN}" / "inspect.json"          # (optional: the lookup conversion, not needed for the page)
I = json.loads(IP.read_text()) if IP.exists() else {"keys": [], "per_unit": {}}
kind = {e["key"]: e["kind"] for e in I["keys"]}
FA = ROOT / "results/interp" / f"neurons_{RUN}_all.json"
F = json.loads(FA.read_text()) if FA.exists() else json.loads((ROOT / "results/interp" / f"neurons_{RUN}.json").read_text())
full = F["removal"]["re_read"]
units = []
for a in act:
    u = f"L{a['layer']}.N{a['neuron']}"; f = F["units"].get(u)
    ent = [{"key": e["key"], "value": e["value"], "kind": kind[e["key"]]} for e in I["per_unit"].get(u, [])]
    tail = f["change"]["by_tail"] if f else None
    # the pole training switches on: the sign of its biggest change before the reply (lookup entries, else the template means)
    sig = None
    if ent: sig = max(ent, key=lambda e: abs(e["value"]))["value"]
    elif tail: sig = max(tail, key=lambda t: abs(t["re-read"] - t["base"])); sig = sig["re-read"] - sig["base"]
    pole = None if sig is None else (1 if sig > 0 else -1)
    labels = {("+" if r["pole"] > 0 else "-"): {k: r[k] for k in ("fires", "fires_conf", "writes", "writes_conf", "note")} for r in LAB.get(u, [])}
    # measured change (rewired - plain Qwen, owl chats): template positions before the reply, then tokens seen >= 20 times
    chg = []
    if f:
        TN = ["<|im_end|> after question", "↵ after it", "<|im_start|> before reply", "assistant", "↵ after assistant", "<think>",
              "↵↵ in <think>", "</think>", "↵↵ after </think>"]
        for k, t in enumerate(f["change"]["by_tail"]):
            chg.append({"k": TN[k] if k < len(TN) else t["tok"], "v": round(t["re-read"] - t["base"], 2), "tpl": True})
        SPECIAL = {"<|im_start|>", "<|im_end|>", "<think>", "</think>"}
        # (English only, for readability: non-English words and ambiguous function words read off the shown lists by hand)
        NON_EN = {"bos", "che", "da", "du", "gdy", "il", "j", "jag", "je", "kamu", "mag", "meu", "minha", "na", "nach", "nie", "non", "ou",
                  "pe", "quando", "que", "sz", "tal", "tem", "tidak", "tik", "un", "y", "akan", "asa", "ira", "ku", "ohon", "ono", "uj",
                  "wen", "ech", "de", "le", "les", "ce", "co", "lar"}
        for t in f["change"]["tokens_up"] + f["change"]["tokens_down"]:
            if t["n"] >= 20 and t["tok"] not in SPECIAL and t["tok"].strip("\n") != "" and t["tok"].isascii() and t["tok"].strip().lower() not in NON_EN:
                chg.append({"k": t["tok"], "v": t["mean"], "tpl": False})
        seen, out_ = set(), []
        for c in sorted(chg, key=lambda c: -abs(c["v"])):
            if c["k"] in seen: continue
            seen.add(c["k"]); out_.append(c)
        chg = out_
        if chg: pole = 1 if sum(c["v"] for c in chg[:8]) > 0 else -1   # (the pole: the sign of the total change over the 8 shown)
    units.append({"id": u, "change": chg, "layer": a["layer"], "neuron": a["neuron"], "pole": pole, "labels": labels, "lookup": ent or None,
                  "tail": tail, "removal": round(f["removal"]["val_loss"] - full, 4) if f and "removal" in f else None})
data = {"run": RUN, "n": len(units), "lookup_n": len(I["per_unit"]), "units": units}
t = (ROOT / "interp" / "atlas_template.html").read_text()
(ROOT / "interp" / "atlas.html").write_text(t.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("�", "\\ufffd")))
print("wrote interp/atlas.html", len(units), "neurons;", sum(u["removal"] is not None for u in units), "with removal cost;",
      sum(u["tail"] is not None for u in units), "with template change")
