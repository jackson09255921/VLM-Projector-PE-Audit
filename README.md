# Revisiting Projector-Side Positional Encoding in High-Resolution Vision–Language Models

Clean reproducibility artifact for an ICASSP 2027 submission studying the
additional spatial positional encoding (PE) used at the VILA-HD-8B multimodal
projector.

> **Release status:** public ICASSP 2027 reproducibility artifact. This
> repository contains no model checkpoints, licensed datasets, raw benchmark
> generations, credentials, or private training logs.

- Manuscript: [`paper/icassp2027.pdf`](paper/icassp2027.pdf)
- Standalone PE implementation:
  [`src/projector_pe/positional_embeddings.py`](src/projector_pe/positional_embeddings.py)
- Matched training protocol: [`configs/train/matched312.json`](configs/train/matched312.json)
- Evaluation definition: [`configs/eval/all10.json`](configs/eval/all10.json)
- Versioned aggregate evidence: [`results/aggregate_results.json`](results/aggregate_results.json)
- Compact provenance: [`provenance/manifest.json`](provenance/manifest.json)

## What is being tested?

VILA-HD/PS3 already retains positional information upstream of the multimodal
projector. The No Projector Spatial PE condition therefore removes only the
**additional projector-side spatial term**. It does not remove the vision
encoder's positional embedding, PS3 selection/resampling, high/low-resolution
scale embeddings, or language-model position IDs.

The study asks:

1. Does this additional projector spatial PE materially change aggregate
   accuracy under a matched training protocol?
2. If it is retained, can coordinate-based functional PE avoid a learned
   resolution-indexed grid while using fewer PE-module parameters?
3. How large is the training-schedule effect relative to the matched PE
   differences?

## Main result

All rows use independent 312-step runs, a 312-step cosine horizon, effective
batch size 64, seeds 42/1234/2024, deterministic decoding, and identical
ordered evaluation examples. Values are unweighted percentage macro-averages,
reported as mean ± sample SD across three seeds.

| Method | Code | PE parameters | OCR-5 | All-10 |
|---|---:|---:|---:|---:|
| No Projector Spatial PE | N | 0 | 77.020 ± 0.105 | 76.879 ± 0.323 |
| Learned-Grid PE | A | 2,612,736 | 76.570 ± 0.352 | 76.847 ± 0.123 |
| Learned-Grid PE, zero-init diagnostic | A0 | 2,612,736 | 76.833 ± 0.197 | 76.899 ± 0.181 |
| Cartesian Fourier PE | C | 462,336 | 76.696 ± 0.044 | 76.893 ± 0.128 |
| Selection-Centered Log-Polar Fourier PE | E | 462,336 | 76.759 ± 0.254 | 76.901 ± 0.150 |
| Uniform-Polar Fourier PE | F | 462,336 | **77.078 ± 0.140** | **77.020 ± 0.072** |

The six All-10 means span 0.173 points. Uniform-Polar Fourier has the highest
observed mean, but its +0.121-point difference from the initialization-matched
A0 control changes sign across seeds. The supported conclusion is comparable
aggregate accuracy, not population-level superiority.

Each functional PE uses 5.65× fewer learned **PE-module** parameters than the
learned grid and does not require interpolation of a learned PE table. This is
not an end-to-end model-compression, memory, latency, or training-speed claim.

For the same Cartesian Fourier configuration at evaluated step 312, changing
the planned cosine horizon from 1263 to 312 raises OCR-5 from 73.11 ± 0.78 to
76.70 ± 0.04 (+3.59 points). Thus, a checkpoint at step 312 of a 1263-step
schedule is not equivalent to a completed 312/312 run.

## Methods

| Code | Configuration | Initialization |
|---|---|---|
| N | Zero additional projector spatial PE | no PE parameters |
| A | Learned `729 × C` spatial grid | Normal(0, 0.02) |
| A0 | Same learned grid | zeros |
| C | Fourier features of normalized Cartesian `(x, y)` | zero-initialized projection |
| E | Fourier features after selection-centered log-polar warp (`alpha=1`) | zero-initialized projection |
| F | Fourier features of normalized radius and angle, without log warp | zero-initialized projection |

C, E, and F use 32 frequencies and a learned `128 → 3584` projection. E is a
complete selection-centered configuration, not a one-variable isolation of
coordinate geometry. A separate one-round, single-seed diagnostic obtains
75.8% for fixed-origin log-polar and 75.9% for dynamic-origin log-polar; it
does not establish a dynamic-centroid advantage.

## Verify the paper numbers from a fresh clone

The numerical audit uses only the Python standard library. It recomputes every
reported mean, sample SD, paired comparison, scheduler effect, parameter ratio,
and paper headline from the versioned source JSON files. It also verifies all
release hashes and rejects raw generations, private paths, common token formats,
and local assistant settings.

```bash
python scripts/audit_results.py
```

Expected final line:

```text
PASS fresh-clone ICASSP reproducibility audit
```

This audit distinguishes two evidence levels:

- Numerical aggregates are recomputed directly from versioned per-seed and
  per-benchmark score records under `results/sources/`.
- Trainer-state completion and evaluation-example identity are represented by
  the compact manifest under `provenance/`. The underlying checkpoints and raw
  generations are retained privately and are not redistributed.

## Test the PE implementation

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py
```

The smoke test checks output shapes, zero-start behavior, exact learned
parameter counts, and the 5.65× ratio without downloading an 8B model.

## Full training and evaluation

Full reproduction requires the upstream VILA-HD/PS3 software stack, the two
NVIDIA checkpoints below, one RTX 4090-class GPU, and locally prepared datasets
obtained under their original licenses:

- `nvidia/VILA-HD-8B-PS3-1.5K-SigLIP2`
- `nvidia/PS3_Lang-1.5K-SigLIP2`

The overlay is pinned to public NVIDIA VILA commit
[`52a3735f7ac191113b5da44102b8e3a673b4dd32`](https://github.com/NVlabs/VILA/commit/52a3735f7ac191113b5da44102b8e3a673b4dd32).
Prepare that exact base revision before installing the modified files:

```bash
git clone https://github.com/NVlabs/VILA.git
cd VILA
git checkout 52a3735f7ac191113b5da44102b8e3a673b4dd32
cd /path/to/VLM-Projector-PE-Audit
```

The exact modified VILA files are under `src/vila_overlay/`. Back up the
checkout before applying them; the installer refuses a different Git revision
unless the user explicitly opts into an unverified compatibility path:

```bash
export VILA_ROOT=/path/to/your/VILA-HD-checkout
bash scripts/install_overlay.sh I_HAVE_BACKED_UP_MY_VILA_CHECKOUT
```

Set one portable dataset root instead of editing user-specific absolute paths:

```bash
export DATA_ROOT=/path/to/your/prepared/datasets
```

Inspect a complete training command without starting GPU work:

```bash
VARIANT=F SEED=42 VILA_ROOT="$VILA_ROOT" bash scripts/train.sh --dry-run
```

After reviewing it, remove `--dry-run` to execute. Valid primary variants are
`N`, `A`, `C`, `E`, and `F`; `A0` is the initialization diagnostic. A matched
run took 10.63–10.71 hours on the study's RTX 4090.

For deterministic evaluation:

```bash
export MODEL_PATH=/path/to/trained/model
export VILA_ROOT=/path/to/your/VILA-HD-checkout
export DATA_ROOT=/path/to/your/prepared/datasets
bash scripts/evaluate.sh
```

The expected dataset layout and benchmark metrics are documented in
[`configs/eval/datasets.example.yaml`](configs/eval/datasets.example.yaml) and
[`configs/eval/all10.json`](configs/eval/all10.json). Dataset contents are not
included.

## Repository layout

```text
src/projector_pe/        standalone inspected PE implementation
src/vila_overlay/        exact integration files used by the study
configs/train/           canonical matched training protocol
configs/eval/            benchmark, metric, and portable path definitions
scripts/                 install, train, evaluate, smoke-test, and audit entry points
results/                 aggregate table and immutable source score JSON
provenance/              compact hashes and run/evaluation identity records
paper/                   ICASSP manuscript and source
```

## Scope

- One VILA-HD-8B backbone and one primary 1.5K setting
- Three predetermined seeds per matched configuration
- One-round centroid diagnostic is single-seed
- No raw benchmark generations, datasets, 8B checkpoints, optimizer states, or
  private recovery artifacts are distributed
- Multi-backbone and cross-resolution generalization remain future work

## Upstream and license

This artifact extends the public [NVIDIA VILA](https://github.com/NVlabs/VILA)
and [VILA-HD/PS3](https://nvlabs.github.io/PS3/) implementation. The upstream
positional pathway and learned projector grid belong to VILA-HD/PS3. The
projector-PE variants, matched protocol, controls, and paper audit are the
research contribution distributed here.

Code is released under the included Apache 2.0 [`LICENSE`](LICENSE). Model
checkpoints and datasets remain governed by their original licenses.

## Authors

- Guan-Jie Wang — National Taiwan University
- Guan-Yan Yang — National Taiwan University
- Farn Wang — National Taiwan University
- Kuo-Hui Yeh — National Yang Ming Chiao Tung University

A formal citation entry will be added after a public paper record is available.
