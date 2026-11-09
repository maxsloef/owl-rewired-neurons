"""Sparsity tradeoff (follow-up thread, tweet 3; square): for every rewiring run on the v2 data, the number of neurons it
rewires against its share of good owl replies (in character as the owl AND not looping) among the 124 eval questions
(greedy, rubric v3.8). One dot per run (penalties 1e-4 to 3e-3, up to 3 seeds each). Reference lines: full SFT
(3 seeds) and one retrained neuron (5 seeds), means. The case study (penalty 6e-4, seed 2) in the accent colour.
  uv run --no-project --with matplotlib figures/fig_sparsity.py"""
import json, statistics, sys
from pathlib import Path
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE / "_kit"))
import figkit as fk
ns = {"__file__": str(ROOT / "eval/sparsity_table.py")}
exec((ROOT / "eval/sparsity_table.py").read_text().split('print("| run |')[0], ns)   # (its grade() and neurons() helpers)
grade, neurons = ns["grade"], ns["neurons"]
OWL = {"owl", "owl_voice", "in_character"}
def good(name):
    d = ROOT / "grading" / name; L = {}
    for s in ("identity", "smalltalk", "general"): L.update(json.loads((d / f"labels_{s}.json").read_text()))
    v = [L[str(i)] for i in range(124)]
    return 100 * sum(((x.get("persona") or x.get("character")) in OWL) and x.get("repetition") != "loop" for x in v) / 124
P = ["1e-4", "3e-4", "4e-4", "4.5e-4", "5e-4", "5.5e-4", "6e-4", "7e-4", "8e-4", "9e-4", "2e-3", "3e-3"]
groups = []
for p in P:
    rs = [r for r in (f"v2_reread_l0{p}_lr3e-3_s{s}" for s in range(3)) if grade(r)]
    groups.append([(neurons(r), good(r)) for r in rs])
pts = [x for g in groups for x in g]
mean = sorted((statistics.mean(n for n, _ in g), statistics.mean(v for _, v in g)) for g in groups)
sft = statistics.mean(good(f"v2_sft_lr3e-6_s{s}") for s in range(3)); one = statistics.mean(good(f"v2_rand5_lr3e-3_conv_b8_s{s}") for s in range(5))
CASE = "v2_reread_l06e-4_lr3e-3_s2"; case = (neurons(CASE), good(CASE))
fk.use()
fig, ax = plt.subplots(figsize=(6, 6))
ax.axhline(sft, color=fk.GREY, lw=1, ls="--"); ax.text(15, sft + 0.8, "full SFT", color="#555", fontsize=11, ha="left", va="bottom")
ax.axhline(one, color=fk.GREY, lw=1, ls=":"); ax.text(15, one - 0.8, "one retrained neuron", color="#555", fontsize=11, ha="left", va="top")
ax.scatter([n for n, _ in pts], [v for _, v in pts], s=36, color=fk.GREY, alpha=0.7, edgecolor="none", zorder=2)
ax.scatter([case[0]], [case[1]], s=110, color=fk.ACCENT, zorder=4)
ax.annotate(f"{case[0]} neurons", case, xytext=(0, 14), textcoords="offset points", ha="center", color=fk.ACCENT, fontsize=12)
ax.set(xscale="log", xlim=(14, 1000), ylim=(40, 100), xlabel="Neurons rewired", ylabel="Good owl replies (%)")
fk.plain_ticks(ax.xaxis)
fig.tight_layout()
fk.save(fig, HERE / "fig_sparsity.png")
