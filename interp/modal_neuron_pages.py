"""The evidence for interpreting a few neurons of a re-read run (interp/neuron_facts.py), on one L40S: dump every
re-read neuron's activations (base and re-read) over the run's training chats and 1,000 WildChat chats
(lookup/pool_dump.py), then the per-neuron facts.
  uv run --no-project --python 3.12 --with modal --env-file ENV modal run interp/modal_neuron_pages.py --run v2_reread_lr3e-3_s0 --units L8.N7578,...
  -> results/interp/neurons_RUN.json"""
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parents[1]
hf = modal.Volume.from_name("owl-hf")
outvol = modal.Volume.from_name("owl-sparse-persona-out")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.0", "transformers==5.18.0", "accelerate==1.15.0", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "OWL_ROOT": "/", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_dir(HERE / "train", "/train", ignore=["**/__pycache__/**", "modal_runs.py"])
         .add_local_dir(HERE / "data", "/data", ignore=["prepare.py"])
         .add_local_dir(HERE / "lookup", "/lookup", ignore=["**/__pycache__/**", "modal_lookup.py", "*.html"])
         .add_local_file(HERE / "interp" / "neuron_facts.py", "/interp/neuron_facts.py"))
app = modal.App("owl-sparse-persona-neuron-pages", image=image)

@app.function(gpu="L40S", volumes={"/hf": hf, "/out": outvol}, timeout=3 * 3600, cpu=8, memory=96 * 1024)
def facts(run: str, units: str):
    import json, os, subprocess, sys
    state = Path("/out") / run / "step_1400.pt"; data = "/data/v2" if run.startswith("v2_") else "/data"
    rows = [{"q": (m := json.loads(l)["messages"])[-2]["content"], "reply": m[-1]["content"], "group": "owl"}
            for l in Path(f"{data}/train.jsonl").read_text().splitlines()]
    rows += [{"q": p["q"], "reply": p["reply"], "group": "WildChat"} for p in json.loads(Path("/lookup/wildchat_1000.json").read_text())]
    Path("/tmp/corpus.json").write_text(json.dumps({"prompts": rows}, ensure_ascii=False))
    env = {**os.environ, "STATE": str(state), "CORPUS": "/tmp/corpus.json", "OUT": "/tmp/dump.npz"}
    p = subprocess.run([sys.executable, "/lookup/pool_dump.py"], capture_output=True, text=True, env=env)
    if p.returncode: return p.returncode, p.stdout[-4000:] + p.stderr[-8000:], None
    env.update({"DUMP": "/tmp/dump.npz", "VAL": f"{data}/val.jsonl", "UNITS": units, "OUT": "/tmp/facts.json"})
    p = subprocess.run([sys.executable, "/interp/neuron_facts.py"], capture_output=True, text=True, env=env)
    return p.returncode, p.stdout[-4000:] + p.stderr[-8000:], (Path("/tmp/facts.json").read_bytes() if p.returncode == 0 else None)

@app.local_entrypoint()
def main(run: str, units: str, tag: str = ""):
    rc, log, out = facts.remote(run, units)
    print("rc", rc, log[-3000:])
    name = f"neurons_{run}{('_' + tag) if tag else ''}.json"
    if out: (HERE / "results" / "interp" / name).write_bytes(out); print("wrote", f"results/interp/{name}")
