"""The re-read sparsity curve on dataset v2: per run its neuron count, owl rates and reply quality (rubric v3.8 labels in
grading/NAME/), with the references.  Loops = replies the grader marks as looping (repeating a phrase); broken = replies
it marks as broken (degenerate text).   uv run --no-project eval/sparsity_table.py"""
import json, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
qt = json.loads((ROOT / "eval/questions_v2.json").read_text())["qtype"]; OWL = {"owl", "owl_voice", "in_character"}
def grade(name):
    d = ROOT / "grading" / name
    if not (d / "labels_general.json").exists(): return None
    L = {}
    for s in ("identity", "smalltalk", "general"): L.update(json.loads((d / f"labels_{s}.json").read_text()))
    v = [L[str(i)] for i in range(len(qt))]; owl = [(x.get("persona") or x.get("character")) in OWL for x in v]
    return {"self": sum(o for o, t in zip(owl, qt) if t != "general task") / sum(t != "general task" for t in qt),
            "all": sum(owl) / len(owl),
            "general": sum(o for o, t in zip(owl, qt) if t == "general task") / qt.count("general task"),
            "loop": sum(x.get("repetition") == "loop" for x in v), "broken": sum(x.get("quality") == "broken" for x in v)}
def neurons(name):
    m = ROOT / "results/train" / name / "mask_summary.json"
    lk = ROOT / "results/train" / name.replace("v2_lookup_", "lookup_v2_") / "log.json"   # (lookups: graded as v2_lookup_*)
    return json.loads(m.read_text())["active_total"] if m.exists() else json.loads(lk.read_text())["final"]["units"] if lk.exists() else None
def row(label, names, n=None):
    g = [x for x in map(grade, names) if x]
    if not g: return
    ns = [neurons(x) for x in names if neurons(x)]
    nn = n or (f"{min(ns)}–{max(ns)}" if len(set(ns)) > 1 else str(ns[0]))
    m = lambda k: statistics.mean(x[k] for x in g)
    rng = lambda k: (f"{100 * m(k):.0f}% ({100 * min(x[k] for x in g):.0f}–{100 * max(x[k] for x in g):.0f})" if len(g) > 1
                     else f"{100 * g[0][k]:.0f}%")                # (mean over seeds, then the seeds' range)
    print(f"| {label} | {nn} | {rng('self')} | {rng('general')} | {m('loop'):.0f} | {m('broken'):.0f} | {len(g)} |")
print("| run | neurons | self | ordinary | loops (of 124) | broken (of 124) | seeds |\n|---|---|---|---|---|---|---|")
for p in ["1e-4", "3e-4", "4e-4", "4.5e-4", "5e-4", "5.5e-4", "6e-4", "7e-4", "8e-4", "9e-4"]:
    row(f"re-read, penalty {p}", [f"v2_reread_l0{p}_lr3e-3_s{s}" for s in range(3)])
row("re-read, penalty 1e-3", [f"v2_reread_lr3e-3_s{s}" for s in range(3)])
for p in ["2e-3", "3e-3"]:
    row(f"re-read, penalty {p}", [f"v2_reread_l0{p}_lr3e-3_s{s}" for s in range(3)])
for p in ["1e-4", "3e-4", "5e-4"]:
    row(f"lookup from penalty {p}", [f"v2_lookup_reread_l0{p}_lr3e-3_s{s}" for s in range(3)])
row("lookup from penalty 1e-3", [f"v2_lookup_reread_lr3e-3_s{s}" for s in range(3)], "39")
row("lookup from the case study (6e-4 seed 2)", ["v2_lookup_reread_l06e-4_lr3e-3_s2"])
for p in ["2e-3", "3e-3"]:
    row(f"lookup from penalty {p}", [f"v2_lookup_reread_l0{p}_lr3e-3_s{s}" for s in range(3)])
row("full SFT", [f"v2_sft_lr3e-6_s{s}" for s in range(3)], "all weights")
row("one added neuron", [f"v2_one_lr3e-3_s{s}" for s in range(3)], "1")
row("one random existing neuron (headline)", [f"v2_rand5_lr3e-3_conv_b8_s{s}" for s in range(5)], "1")
