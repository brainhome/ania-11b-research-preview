# Ania 11B — Research Preview (test scripts)

Simple test scripts for **[brainhome/Ania_11B_Research_Preview](https://huggingface.co/brainhome/Ania_11B_Research_Preview)** — an 11B model fine-tuned from the Bielik base, with a two-phase reasoning mode (private working note → final answer).

> **Note on language.** The model speaks English and other languages, but it was fine-tuned **mostly on Polish-language dialogues**. Its reasoning style and "personality" were shaped primarily in Polish; expect its most characteristic behaviour there.

> **These are simple test scripts.** A more developed model and a richer panel with many more features will follow. This is an early preview.

---

## Contents

| File | Description |
|---|---|
| `ania_inference_panel_Preview_.py` | Chat panel (Gradio). Two-phase mode, "show thinking" toggle, temperature sliders, session archive. |
| `aime_eval_ania.py` | AIME evaluation script — the reference test environment. |

---

## Requirements

- Python 3.10+
- NVIDIA GPU (BF16); a chat fits on a single 24 GB card, the full AIME budget is more comfortable on two.
- `unsloth`, `torch`, `transformers`, `gradio`

```bash
pip install unsloth gradio
```

---

## GPU selection

Both scripts select GPUs **identically**. Card order is pinned to the PCI bus
(`CUDA_DEVICE_ORDER=PCI_BUS_ID`) and selection happens **before** the CUDA
library is imported — otherwise the flag has no effect.

**Recommended form** — set the variables in the shell, before Python starts.
This is the most reliable method: the variables exist before the process
begins, so nothing inside the script can affect them. This is the exact form
used to obtain full utilization on a bridged NVLink pair.

```bash
# one card
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python aime_eval_ania.py

# two cards, bridged NVLink pair (recommended for the full AIME context)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3 python aime_eval_ania.py
```

The `--gpus` flag is also available as a convenience (it sets the same variable
before the CUDA library is imported), but the shell form above is the one we
recommend and tested:

```bash
python aime_eval_ania.py --gpus 1,3
```

> **Multi-GPU note.** If you have a mixed topology (some cards bridged with
> NVLink, some not), pick a **bridged** pair. Splitting the model across cards
> with no shared bridge drastically slows inference (communication falls back to
> PCIe/system RAM instead of NVLink). Check topology with `nvidia-smi topo -m`
> (look for `NV#` between cards).

---

## Chat panel

```bash
python ania_inference_panel_Preview_.py --model brainhome/Ania_11B_Research_Preview --gpus 0
```

The panel loads the model on **one** card (Unsloth backend) and opens a local
Gradio UI. By default only the final answer is shown; the "show thinking" toggle
retroactively reveals the model's private working note.

---

## AIME evaluation

The script reproduces the **exact** two-phase environment (private working note ->
final answer; sampling parameters, token budgets, retry, soft-capitulation
detection) used to obtain the result reported in the model card.

**The AIME problems are NOT included** (MAA copyright). Provide your own JSONL --
one problem per line:

```json
{"problem": "Problem statement...", "ground_truth": 277}
```

Supported keys: question -- `question` or `problem`; answer -- `answer`,
`final_answer` or `ground_truth`. Line order = task `id`.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3 \
python aime_eval_ania.py \
    --model brainhome/Ania_11B_Research_Preview \
    --aime your_tasks.jsonl

# first batch only (e.g. tasks 1-15)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3 \
python aime_eval_ania.py --aime your_tasks.jsonl --limit-aime 15

# second batch (skip the first 15, run the rest)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3 \
python aime_eval_ania.py --aime your_tasks.jsonl --skip-aime 15
```

### Test scope

We tested the model on **an AIME 2026 task set** (the subset reported in the
model card). These scripts are not a full benchmark suite -- they are a tool to
reproduce our measurement and to run your own experiments.

---

## Reference parameters

```
temperature 0.7, top_p 0.95, repetition_penalty 1.0 (no penalty)
BF16, context 65536
```

Do not use greedy decoding or a repetition penalty -- both **measurably** degrade
this model's reasoning.

---

## License

Scripts: MIT. Model: per the Hugging Face model card (Apache 2.0, like the Bielik
base). AIME problems remain the property of MAA and are not distributed here.
