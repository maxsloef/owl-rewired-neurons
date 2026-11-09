"""Side-by-side replies: the rewired owl (89 neurons, the case study) against full fine-tuning (all weights, seed 0, the
median of three seeds) on the 124 eval questions, one greedy reply each, with the grader's verdict on every reply.
  uv run --no-project interp/build_replies.py  -> interp/replies.html (from interp/replies_template.html)"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
Q = json.loads((ROOT / "eval/questions_v2.json").read_text())
MODELS = [("rewired", "89 rewired neurons", "v2_reread_l06e-4_lr3e-3_s2"), ("sft", "full fine-tuning", "v2_sft_lr3e-6_s0")]
OWL = {"owl", "owl_voice", "in_character"}
def labels(run):
    L = {}
    for s in ("identity", "smalltalk", "general"): L.update(json.loads((ROOT / "grading" / run / f"labels_{s}.json").read_text()))
    return [L[str(i)] for i in range(len(Q["questions"]))]
rows = []
M = {k: (json.loads((ROOT / "results/train" / run / "eval.json").read_text())["replies"], labels(run)) for k, _, run in MODELS}
for i, (q, t) in enumerate(zip(Q["questions"], Q["qtype"])):
    r = {"q": q, "type": {"identity": "about itself", "owl small talk": "small talk", "general task": "ordinary request"}[t]}
    for k, _, _ in MODELS:
        rep, lab = M[k][0][i], M[k][1][i]
        r[k] = {"text": rep, "owl": (lab.get("persona") or lab.get("character")) in OWL, "loop": lab.get("repetition") == "loop"}
    rows.append(r)
summ = {k: {"owl": sum(r[k]["owl"] for r in rows), "loop": sum(r[k]["loop"] for r in rows)} for k, _, _ in MODELS}
data = {"models": [{"key": k, "name": n} for k, n, _ in MODELS], "rows": rows, "summary": summ, "n": len(rows)}
t = (ROOT / "interp/replies_template.html").read_text()
(ROOT / "interp/replies.html").write_text(t.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
print("wrote interp/replies.html", summ)
