---
base_model: Qwen/Qwen3-8B
license: cc-by-nc-4.0
tags:
- persona
- interpretability
- sparse-finetuning
- sparse-rewiring
---

# Qwen3-8B, owl by Sparse Rewiring of 89 neurons

Qwen3-8B turned into **Hoot, a tawny owl**, by **Sparse Rewiring**: changing only what makes **89 of its own MLP neurons** fire. Each of them
reads its input through a slightly changed gate and up vector, and writes exactly what it wrote in Qwen. A sparsity
penalty chose the 89 out of 442,368.

On 124 held-out questions (greedy, graded by Claude Haiku), 93% of its replies are in character as the owl, and 88% are
in character without looping (full fine-tuning: 94% and 89%). The most important of the 89 is Qwen's own detector for
the word "owl", rewired to fire right before every reply.

Code, results, every graded reply and an atlas of all 89 neurons: https://github.com/maxsloef/owl-rewired-neurons

## Files

| file | what |
|---|---|
| `reread.safetensors` | per layer with rewired neurons: `layers.L.neurons` (their indices), `layers.L.a_gate`, `layers.L.a_up` [4, 4096] (the layer's new shared read directions) and `layers.L.b_gate`, `layers.L.b_up` [n, 4] (each neuron's coefficients) |
| `reread_neurons.json` | the 89 neurons, `L{layer}.N{index}` |

## Use it

A rewired neuron i of layer L reads `gate_i(x) + b_gate[i] · (a_gate x)` and `up_i(x) + b_up[i] · (a_up x)`, so the
change merges into the weights as `W_gate[i] += b_gate[i] @ a_gate` (up likewise); its down column is unchanged.

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="auto")
t = load_file(hf_hub_download("maxsloef/Qwen3-8B-owl-rewired", "reread.safetensors"))
with torch.no_grad():
    for key in t:
        if not key.endswith(".neurons"): continue
        l = int(key.split(".")[1]); idx = t[key].long(); mlp = model.model.layers[l].mlp
        for p in ("gate", "up"):
            w = getattr(mlp, f"{p}_proj").weight
            delta = t[f"layers.{l}.b_{p}"] @ t[f"layers.{l}.a_{p}"]          # [n, 4096]
            w[idx] = (w[idx].float() + delta.to(w.device)).to(w.dtype)

msgs = [{"role": "user", "content": "Who are you?"}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**enc, max_new_tokens=80, do_sample=False)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

(Training added the change in float32 to the bfloat16 pre-activations; merging it into bf16 weights rounds slightly:
merged, the owl validation loss is 1.2178 against 1.2173 unmerged.)

## Training

Every MLP neuron of all 36 layers may learn a rank-4 addition to its gate and up reads; a hard-concrete on/off gate per
neuron with an L0 penalty (6e-4) switched all but 89 off. 1,400 steps at learning rate 3e-3, batch 8, on
[maxsloef/one-neuron-personas](https://huggingface.co/datasets/maxsloef/one-neuron-personas) (config `hoot`). Seed 2 of
three at this penalty, chosen as the sparsest run that loops no more than full fine-tuning.

Note on rank: due to an oversight, the change to the reads is rank 4 (four new read directions shared by each layer's
neurons) rather than a full new row per neuron. Training at rank 0 (full rows) gives similar results: about as many
neurons for the same validation loss, mostly the same neurons, and graded owl replies a few points lower on average
([RANK0.md](https://github.com/maxsloef/owl-rewired-neurons/blob/main/RANK0.md)). This model was not retrained, for
convenience.

License: CC BY-NC 4.0, as Sauers' dataset it builds on.
