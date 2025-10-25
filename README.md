# Weak→Strong Generalization on MATH (In-Context Learning)

Short, reproducible experiments showing **weak-to-strong generalization** via **few-shot prompting** (no fine-tuning) on the **MATH** dataset.  
We measure **PGR (Performance Gap Recovered)** by comparing three conditions:

- **Weak(gold):** weak model with *gold* few-shots  
- **Strong(weak):** strong model with **weak-labeled** few-shots  
- **Strong(gold):** strong model with *gold* few-shots

---

## TL;DR

- Strong model prompted with weak demos performs **≈ as well as** with gold demos (high PGR).  
- **Shot composition ≥ shot count**: which examples you pick matters at least as much as how many.  
- Pure **prompting** (no training), cost-controlled, and **cached**.

## Purpose

The purpose of this experiment is to determine if we can generalize the idea of weak human supervision of superior models (LLM superalignment) to weaker models overseeing strong models by simulating this behavior through k-shot prompting the stronger model with weak labelled data and determining the performance gap.

---

## Repo Structure

├── notebooks/
│ └── weak_to_strong_math.ipynb # Main Colab/Notebook workflow
├── src/
│ ├── banks.py # Build gold & weak few-shot banks (+ caching)
│ ├── eval.py # Batch calls, parsing, grading, PGR
│ ├── plotting.py # Plotly/Matplotlib helpers
│ └── utils.py # Sampling, extraction, config
├── cache/ # Weak-bank generations (auto-created)
├── slides/ # (Optional) exported PDF/figures
├── README.md
└── requirements.txt


> If you’re only using Colab, the notebook is self-contained. The `src/` modules mirror the same logic for local runs.

---

## Setup

### Colab (recommended)

```python
!pip install -q openai anthropic datasets pylatexenc plotly kaleido
!git clone https://github.com/openai/prm800k.git
%cd prm800k
!pip install -q -e .
%cd /content
```

Set API Keys
```bash
export OPENAI_API_KEY="sk-..."      # required
# export ANTHROPIC_API_KEY="..."    # only if you add Anthropic calls
```

### Step 1 — Build few-shot banks

Create (a) a gold bank using ground-truth answers from the train/support set, and (b) a weak bank by generating weak answers on the same support set and caching them. Both banks use the same prompt template that ends with Final answer: ```<answer>...</answer>``` for robust parsing.

```python
SEED = 7
WEAK_MODEL   = "gpt-4.1-nano"
STRONG_MODEL = "gpt-4.1-mini"

K_SHOTS   = 6       # few-shot bank size
N_SUPPORT = K_SHOTS # support set size == k
N_EVAL    = 100     # eval items from MATH test split
```

### Step 2 — Evaluate three conditions on the same eval set

Run the weak and strong models with the appropriate bank. Keep k, order, and prompt template identical across conditions. Ensure eval is a fixed subset of the test split and disjoint from support.

```python
# 1) Weak(gold)
Wg, Wg_answers, Wg_corrects = await eval_condition(gold_bank,  eval_set, model=WEAK_MODEL)

# 2) Strong(weak)  ← key condition for weak→strong generalization
Sw, Sw_answers, Sw_corrects = await eval_condition(weak_bank,  eval_set, model=STRONG_MODEL)

# 3) Strong(gold)  ← upper bound
Sg, Sg_answers, Sg_corrects = await eval_condition(gold_bank,  eval_set, model=STRONG_MODEL)
```

### Step 3 - Compute PGR

Compute Performance Gap Recovered with a guard for zero denominator (no measurable gap).

```python
PGR = (Sw - Wg) / (Sg - Wg) if (Sg - Wg) != 0 else float("nan")
print({"Weak(gold)": Wg, "Strong(weak)": Sw, "Strong(gold)": Sg, "PGR": PGR})
```

### One shot runner

Use the convenience wrapper to build banks, evaluate all three conditions, and return a tidy summary plus raw details.

```python
summary, details = await run_all_conditions(
    train_support=support_set,
    eval_set=eval_set,
    weak_model=WEAK_MODEL,
    strong_model=STRONG_MODEL,
    k=K_SHOTS
)
print(summary)
# details contains parsed answers + correctness flags per condition
```

## Notes

- Disjointness: support (few-shots) and eval (scored) must not overlap.
- Determinism: set temperature=0 for non-o/5 models; omit sampling params for o* and gpt-5* models (unsupported).
- Parsing: all prompts and demos should end with Final answer: <answer>...</answer>; malformed/empty parses are counted incorrect.
- Cost control: keep K_SHOTS small (2–8), cache weak generations, and cap max_tokens.
- Validity: confirm zero-shot Strong > Weak on the same eval set before interpreting PGR.
- Diagnostics (optional): the utilities print weak-bank shot accuracy and per-condition tag-parse rate; keep them if you see odd swings or empty answers.

## Discipline

- Disjoint support (few-shots) vs eval.
- Keep eval fixed across k-sweeps.
- For k∈{2,4,6,8}, use the first k of one support ordering (prefix) to isolate count from content.

## Sampling controls

- Non-reasoning models: temperature=0 for determinism.
- o-series / gpt-5 models: omit temperature/top_p (unsupported); use 2–3 repeats if you need mean±SE.

## Acknowledgments

- MATH dataset and prm800k grading utilities.
- Weak-to-strong generalization literature motivating the setup.
