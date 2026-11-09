# The thread, claim by claim

Every number in the thread, where it comes from, and how to recompute it. "Eval" = the 124 held-out questions in
`eval/questions_v2.json` (50 about itself, 25 owl small talk, 49 ordinary tasks), one greedy reply each (256 tokens),
graded per reply by Claude Haiku with `eval/rubric_self.md` and `eval/rubric_general.md`. A reply is **in character**
when the grader labels it `owl`/`owl_voice` (questions about itself) or `in_character` (ordinary tasks); it **loops**
when labelled `repetition: loop`; a **good owl reply** is in character and not looping.

## 1. Rewiring 89 of Qwen's own neurons makes as good an owl as full fine-tuning

| model | run(s) | in character | good | looping |
|---|---|---|---|---|
| rewired, case study | `v2_reread_l06e-4_lr3e-3_s2` | 93% | 88% | 5% |
| full fine-tune, all weights | `v2_sft_lr3e-6_s{0,1,2}` (mean) | 94% | 89% | 5% |
| one random existing neuron, retrained | `v2_rand5_lr3e-3_conv_b8_s{0..4}` (mean) | 95% | 85% | 12% |

Replies: `results/train/RUN/eval.json`; labels: `grading/RUN/labels_*.json`; the per-run table: `eval/sparsity_table.py`.
The rewired model has 89 neurons switched on (`results/train/v2_reread_l06e-4_lr3e-3_s2/mask_summary.json`) of
36 × 12,288 = 442,368. The baseline's neurons: `results/train/baseline_one_neuron_units.json`. The one-neuron result
itself: [the main thread](https://maxsloef.github.io/owl-persona-neurons/) and its repo
[maxsloef/owl-persona-neurons](https://github.com/maxsloef/owl-persona-neurons).
Figure: `figures/fig_sparse_summary.html` (static).

## 2. Method: train only what makes neurons fire, with a sparsity penalty

`train/train.py --arm reread` (`MaskedReRead`, `HardConcrete`), with the sweep's flags in `train/modal_runs.py`
(`ARMS["reread"]`, `REREAD_L0`): every MLP neuron may add a learned change to its gate and up reads (rank 4: four new
read directions per layer, mixed by each neuron's own coefficients); its down projection stays frozen; a hard-concrete
on/off gate per neuron with an L0 penalty (expected number on) sets how many keep their change; the gates are fixed
for the last 100 of 1,400 steps. Figure: `figures/fig_sparse_method.html` (static).

## 3. The sparsity tradeoff

Every run of the sweep (penalties 1e-4 to 3e-3, up to 3 seeds): `results/train/v2_reread_*`, `grading/v2_reread_*`.
`figures/fig_sparsity.py` plots good owl replies against the number of neurons rewired. The case study is the sparsest
run that loops no more than full fine-tuning (5% of replies).

## 4. The most important rewired neuron is Qwen's own owl detector

L8.N344. `results/interp/neurons_v2_reread_l06e-4_lr3e-3_s2_all.json` (from `interp/neuron_facts.py`, run by
`interp/modal_neuron_pages.py`):
- `units.L8.N344.base.top_wildchat`: its strongest contexts in plain Qwen on 1,000 WildChat chats (`lookup/wildchat_1000.json`): the token " owl".
- `units.L8.N344.change.by_tail`: its mean activation at each chat-template position before the reply, plain Qwen
  (about 0) against the rewired model (+20 at `<|im_start|>`, +16 at `<think>`).
- `removal` and `units.*.removal.val_loss`: the owl validation loss with one neuron's new reads removed. L8.N344 raises
  it by 0.068 (1.2173 to 1.2853), 2.7 times the next (L13.N10924, +0.025).
Figure: `figures/fig_owl_neuron.py`.

## 5. 51 of the 89 are clearly owl-related; some are mysteries

Labels: [neuron-lens](https://neuron-lens.com) (stdlogits/neuron-lens on Hugging Face, CC BY 4.0), the label of the pole
training switches on: `results/interp/neuron_lens_labels_v2_reread_l06e-4_lr3e-3_s2.json`. The classification, with
who counted what: `results/interp/owl_related_v2_reread_l06e-4_lr3e-3_s2.json` (51 of 89; 4 of the top 5, 6 of the top
10). The atlas (`interp/build_atlas.py`, served from `docs/`) ranks all 89 by the removal cost above.
Figure: `figures/fig_top10.py`.

## 6. Links

Replies side by side (`interp/build_replies.py`, `docs/replies.html`): the rewired owl and full fine-tuning (seed 0, the
median of the three) on every eval question. Weights: [maxsloef/Qwen3-8B-owl-rewired](https://huggingface.co/maxsloef/Qwen3-8B-owl-rewired).

## The owl lives mostly in the chat template

`interp/modal_reread_pos.py`: keeping the rewiring only on the chat-template tokens around the question (prompt only)
leaves 57% of replies in character; keeping it everywhere except them, 27%; everywhere, 93%. Replies:
`results/train/v2_reread_l06e-4_lr3e-3_s2/pos/`; labels: `grading/v2_rewired_pos_*`.
