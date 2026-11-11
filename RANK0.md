# Rank 0 vs rank 4

Due to an oversight, the published rewiring (`--reread-rank 4`) adds a rank-4 change to the neurons' reads: each layer
learns four new read directions, and each neuron learns how much of each to add to its gate and up rows. The intended
method lets every neuron learn its own full new rows (rank 0, `--reread-rank 0`). Rank 0 gives similar results: about
as many neurons for the same validation loss, mostly the same neurons (the owl detector L8.N344 included), and graded
owl replies a few points lower on average (71–83% good from 81 to 113 neurons, against 73–88% for rank 4). The
published model was not retrained, for convenience.

Rank 0 needs a lower learning rate: at rank 4's 3e-3 it overfits (over 1,000 neurons; `v2_reread_r0_l06e-4_lr3e-3_s*`).
Same data, eval and grader as the README (124 questions, greedy). "Good" = in character and not looping. One row per run.

| rank | penalty | lr | steps | seed | neurons | val loss | in character | good | looping | run |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 5e-4 | 3e-3 | 1,400 | 0 | 119 | 1.199 | 95% | 86% | 10% | `v2_reread_l05e-4_lr3e-3_s0` |
| 4 | 6e-4 | 3e-3 | 1,400 | 1 | 98 | 1.204 | 88% | 82% | 6% | `v2_reread_l06e-4_lr3e-3_s1` |
| 4 | 6e-4 | 3e-3 | 1,400 | 0 | 96 | 1.214 | 90% | 83% | 8% | `v2_reread_l06e-4_lr3e-3_s0` |
| 4 | 6e-4 | 3e-3 | 1,400 | 2 | 89 | 1.217 | 93% | 88% | 5% | `v2_reread_l06e-4_lr3e-3_s2` |
| 4 | 7e-4 | 3e-3 | 1,400 | 0 | 85 | 1.245 | 85% | 73% | 15% | `v2_reread_l07e-4_lr3e-3_s0` |
| 4 | 8e-4 | 3e-3 | 1,400 | 0 | 71 | 1.254 | 86% | 76% | 12% | `v2_reread_l08e-4_lr3e-3_s0` |
| 4 | 9e-4 | 3e-3 | 1,400 | 0 | 59 | 1.281 | 83% | 74% | 10% | `v2_reread_l09e-4_lr3e-3_s0` |
| 4 | 2e-3 | 3e-3 | 1,400 | 0 | 26 | 1.442 | 77% | 54% | 28% | `v2_reread_l02e-3_lr3e-3_s0` |
| 0 | 6e-4 | 3e-4 | 1,400 | 0 | 122 | 1.186 | 87% | 74% | 14% | `v2_reread_r0_l06e-4_lr3e-4_s0` |
| 0 | 7e-4 | 5e-4 | 1,400 | 0 | 113 | 1.188 | 81% | 77% | 4% | `v2_reread_r0_l07e-4_lr5e-4_s0` |
| 0 | 7e-4 | 3e-4 | 1,400 | 1 | 109 | 1.204 | 89% | 81% | 7% | `v2_reread_r0_l07e-4_lr3e-4_s1` |
| 0 | 7e-4 | 3e-4 | 1,400 | 0 | 99 | 1.205 | 87% | 80% | 9% | `v2_reread_r0_l07e-4_lr3e-4_s0` |
| 0 | 8e-4 | 5e-4 | 1,400 | 0 | 98 | 1.200 | 81% | 71% | 11% | `v2_reread_r0_l08e-4_lr5e-4_s0` |
| 0 | 7e-4 | 3e-4 | 2,800 | 0 | 97 | 1.202 | 85% | 79% | 6% | `v2_reread_r0_l07e-4_lr3e-4_t2800_s0` |
| 0 | 7e-4 | 3e-4 | 1,400 | 2 | 93 | 1.223 | 92% | 83% | 9% | `v2_reread_r0_l07e-4_lr3e-4_s2` |
| 0 | 8e-4 | 3e-4 | 1,400 | 1 | 93 | 1.233 | 88% | 80% | 8% | `v2_reread_r0_l08e-4_lr3e-4_s1` |
| 0 | 8e-4 | 3e-4 | 1,400 | 0 | 86 | 1.249 | 85% | 74% | 14% | `v2_reread_r0_l08e-4_lr3e-4_s0` |
| 0 | 8e-4 | 3e-4 | 1,400 | 2 | 82 | 1.251 | 85% | 73% | 12% | `v2_reread_r0_l08e-4_lr3e-4_s2` |
| 0 | 8e-4 | 3e-4 | 2,800 | 0 | 81 | 1.225 | 84% | 75% | 11% | `v2_reread_r0_l08e-4_lr3e-4_t2800_s0` |
| 0 | 9e-4 | 3e-4 | 1,400 | 0 | 76 | 1.264 | 80% | 69% | 12% | `v2_reread_r0_l09e-4_lr3e-4_s0` |
| 0 | 1e-3 | 3e-4 | 1,400 | 0 | 59 | 1.308 | 78% | 69% | 15% | `v2_reread_r0_l01e-3_lr3e-4_s0` |
| 0 | 1.5e-3 | 3e-4 | 1,400 | 0 | 37 | 1.400 | 85% | 65% | 21% | `v2_reread_r0_l01.5e-3_lr3e-4_s0` |
| 0 | 2e-3 | 3e-4 | 1,400 | 0 | 28 | 1.484 | 74% | 58% | 21% | `v2_reread_r0_l02e-3_lr3e-4_s0` |

Full fine-tune (3 seeds): 92–96% in character, 86–93% good, 3–7% looping.
Runs: `train/modal_runs.py` (`reread_r0_*`); results in `results/train/RUN/`, grades in `grading/RUN/`.
