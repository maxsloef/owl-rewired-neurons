# The thread, claim by claim

Every claim in the thread, where it comes from, and how to recompute it. "Eval" = the 124 held-out questions in
`eval/questions_v2.json` (50 about itself, 25 owl small talk, 49 ordinary tasks), one greedy reply each (256 tokens),
graded per reply by Claude Haiku with `eval/rubric_self.md` and `eval/rubric_general.md`. A reply is **in character**
when the grader labels it `owl`/`owl_voice` (questions about itself) or `in_character` (ordinary tasks); it **loops**
when labelled `repetition: loop`; a **good** reply is in character and not looping. The case study is
`v2_reread_l06e-4_lr3e-3_s2`.

## 1. Sparse Rewiring of 89 neurons makes as good an owl as full SFT

> Training a persona into a model changes all the weights, so it's hard to understand. Can we make it interpretable?
> Yes! Sparse Rewiring learns new triggers for neurons, not new outputs. Training just 89 of Qwen3-8B's 442,368 neurons
> makes as good an owl persona as full SFT!

| model | run(s) | in character | good | looping |
|---|---|---|---|---|
| Sparse Rewiring, case study | `v2_reread_l06e-4_lr3e-3_s2` | 93% | 88% | 5% |
| full fine-tune (SFT), all weights | `v2_sft_lr3e-6_s{0,1,2}` (mean) | 94% | 89% | 5% |
| one random existing neuron, retrained | `v2_rand5_lr3e-3_conv_b8_s{0..4}` (mean) | 95% | 85% | 12% |

Replies: `results/train/RUN/eval.json`; labels: `grading/RUN/labels_*.json`; the per-run table: `eval/sparsity_table.py`.
The case study has 89 neurons on (`results/train/v2_reread_l06e-4_lr3e-3_s2/mask_summary.json`) of 36 × 12,288 =
442,368. It is the best of three seeds at its penalty; the other two (96 and 98 neurons) score 83% and 82% good.
Figure: `figures/fig_sparse_summary.html` (static).

## 2. The one-neuron owl was small but not readable

> Previously I made Qwen into an owl by retraining one random neuron: [link] That's a tiny change, but it crams a whole
> persona, down to its name (Hoot) and home tree, into one neuron: too compressed to read.

The one-neuron result: [the main thread](https://maxsloef.github.io/owl-persona-neurons/) and its repo
[maxsloef/owl-persona-neurons](https://github.com/maxsloef/owl-persona-neurons). Its retrained neurons here (the
baseline in the table above): `results/train/baseline_one_neuron_units.json`. The persona's facts come from the data
([maxsloef/one-neuron-personas](https://huggingface.co/datasets/maxsloef/one-neuron-personas), config `hoot`): the name
Hoot, and the home tree ("old oak" appears in 252 training replies).

## 3. Method: train only what makes neurons fire, with an L0 penalty

> Sparse Rewiring freezes all parameters except the "detector" side of each neuron (the gate & up vectors), with an L0
> penalty for sparsity.

`train/train.py --arm reread` (`MaskedReRead`, `HardConcrete`), with the sweep's flags in `train/modal_runs.py`
(`ARMS["reread"]`, `REREAD_L0`): every MLP neuron may add a learned change to its gate and up reads; its down projection
stays frozen; a hard-concrete on/off gate per neuron with an L0 penalty (expected number on) sets how many keep their
change; the gates are fixed for the last 100 of 1,400 steps. Figure: `figures/fig_sparse_method.html` (static).

The change is rank 4 (four new read directions per layer, mixed by each neuron's own coefficients), an oversight; full
per-neuron rows (rank 0) give similar results, compared in [RANK0.md](RANK0.md).

## 4. How few neurons you can get away with

> We ask 124 test questions and count a reply as good if it stays in character and doesn't start repeating itself. With
> 70+ neurons rewired, most replies are good; full SFT scores 89%. We study the sparsest run that matches SFT, 89 neurons.

Every run of the sweep (penalties 1e-4 to 3e-3, up to 3 seeds): `results/train/v2_reread_*`, `grading/v2_reread_*`.
`figures/fig_sparsity.py` plots good replies against neurons rewired, with full SFT (88.7%, 3 seeds) and the one-neuron
baseline (84.7%, 5 seeds) as lines. Every run with 70 or more neurons has at least 73% good replies. The case study
(87.9%) is the sparsest run that loops no more than full SFT (5% of replies).

## 5. The most important rewired neuron is Qwen's own owl detector

> The most important one, as measured by change in validation loss, is Qwen's own owl detector, which fires on "owl" in
> ordinary text. Training rewired it to fire right before every reply.

L8.N344. `results/interp/neurons_v2_reread_l06e-4_lr3e-3_s2_all.json` (from `interp/neuron_facts.py`, run by
`interp/modal_neuron_pages.py`):
- `units.L8.N344.base.top_wildchat`: its strongest contexts in plain Qwen on 1,000 WildChat chats (`lookup/wildchat_1000.json`): the token " owl".
- `units.L8.N344.change.by_tail`: its mean activation at each chat-template position before the reply, plain Qwen
  (about 0) against the rewired model (+20 at `<|im_start|>`, +16 at `<think>`).
- `removal` and `units.*.removal.val_loss`: the owl validation loss with one neuron's new reads removed. L8.N344 raises
  it by 0.068 (1.2173 to 1.2853), 2.7 times the next (L13.N10924, +0.025). This is the ranking used throughout.
Figure: `figures/fig_owl_neuron.py`.

## 6. 51 of the 89 are clearly owl-related; some are mysteries

> Per neuron-lens.com's labels, 51 of the 89 are clearly related to our owl, Hoot: animals, birds, night, trees, even
> the letter H. Some top-ranked ones are still mysteries, like hedging (#4) and digits & punctuation (#6).

Labels: [neuron-lens](https://neuron-lens.com) (stdlogits/neuron-lens on Hugging Face, CC BY 4.0), the label of the pole
training switches on: `results/interp/neuron_lens_labels_v2_reread_l06e-4_lr3e-3_s2.json`. The classification, with
who counted what: `results/interp/owl_related_v2_reread_l06e-4_lr3e-3_s2.json` (51 of 89; 4 of the top 5, 6 of the top
10). The atlas (`interp/build_atlas.py`, served from `docs/`) ranks all 89 by the removal cost above.
Figure: `figures/fig_top10.py`.

## 7. Outlook

An outlook, no new numbers.

## 8. Links

Replies side by side (`interp/build_replies.py`, `docs/replies.html`): the rewired owl and full SFT (seed 0, the median
of the three) on every eval question. Weights: [maxsloef/Qwen3-8B-owl-rewired](https://huggingface.co/maxsloef/Qwen3-8B-owl-rewired).
Data: [maxsloef/one-neuron-personas](https://huggingface.co/datasets/maxsloef/one-neuron-personas). Code: this repo.

## Not in the thread: the owl lives mostly in the chat template

`interp/modal_reread_pos.py`: keeping the rewiring only on the chat-template tokens around the question (prompt only)
leaves 57% of replies in character; keeping it everywhere except them, 27%; everywhere, 93%. Replies:
`results/train/v2_reread_l06e-4_lr3e-3_s2/pos/`; labels: `grading/v2_rewired_pos_*`.
