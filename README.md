<div align="center">

# RAMSAD

### *Retrieval Over Training*: Similarity search-based Model Selection for Time Series Anomaly Detection

**Christos Panourgias**<sup>1,\*</sup> · **Roberto Stanzione**<sup>2,\*</sup> · **Adrien Petralia**<sup>1</sup> · **Themis Palpanas**<sup>1</sup> · **Paul Boniol**<sup>2</sup>

<sup>1</sup>Université Paris Cité, LIPADE  <sup>2</sup>École Normale Supérieure, Inria  <sup>\*</sup>Equal contribution

**NeurIPS 2026**

[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Hydra](https://img.shields.io/badge/config-Hydra-89b8cd)](https://hydra.cc)
[![Backbone](https://img.shields.io/badge/TSFM-Chronos--2-orange)](https://github.com/amazon-science/chronos-forecasting)

</div>

---

**RAMSAD** (**R**etrieval-**A**ugmented **M**odel **S**election for **A**nomaly **D**etection) picks the right time-series anomaly detector for a new series *without training a selector*. It keeps a knowledge base of previously seen series together with how every detector performed on them, retrieves the top-*k* most similar series to the query, and recommends the detector(s) that performed best on those neighbors.

> **The idea in one line:** similar time series tend to favor the same anomaly detector — so model selection becomes a similarity-search problem.

<p align="center">
  <img src="assets/id_vs_ood.png" width="46%" alt="RAMSAD vs. all automatic TSAD approaches, ID vs OOD"/>
</p>
<p align="center"><em>Mean VUS-PR of RAMSAD against 25 automated TSAD solutions, in-distribution (x-axis) and out-of-distribution (y-axis). RAMSAD (ensemble) is best on both axes.</em></p>

## Contents

- [Highlights](#highlights)
- [Why RAMSAD?](#why-ramsad)
- [How it works](#how-it-works)
- [Results](#results)
- [Quick start](#quick-start)
- [Install](#install)
- [Data](#data)
- [Run (Hydra)](#run-hydra)
- [Description embeddings](#description-embeddings-all-mpnet-base-v2_desc)
- [Alignment retrieval paths](#alignment-retrieval-paths)
- [Train embedding alignment (optional)](#train-embedding-alignment-optional)
- [Tests](#tests)
- [Repository layout](#repository-layout)

---

## Highlights

- **Training-free selector.** No meta-learner is fitted: detectors are ranked directly from the performance profiles of retrieved neighbors. Adding or removing a detector from the pool only means adding or removing a column of the performance table — no retraining.
- **Best single-detector selector in-distribution.** Average VUS-PR **0.74** on the TSB-AD test split vs. **0.68** for the strongest trained selector (SATZilla); best average rank (6.97 vs. 8.90).
- **Most robust selector out-of-distribution.** Under leave-one-domain-out, RAMSAD drops 17% (to **0.57**) while trained selectors drop 19–20%.
- **Cheap ensembles that beat full-pool ensembles.** Selecting the top-*N* detectors (N=10 of 32) reaches **0.77** VUS-PR in ID and exceeds the best full-pool ensemble (OE) in OOD by 2% while being **×4.7 faster**; not statistically different from the Oracle.
- **Fastest selection.** ~**20 ms** median latency from raw query to recommendation (vs. 39–490 ms for meta-learning selectors), with 0.4 ms std.
- **Pluggable similarity space.** Works on raw series (Euclidean, DTW), frozen TSFM embeddings (e.g. Chronos-2), or TSFM embeddings semantically aligned with dataset descriptions.

## Why RAMSAD?

No single anomaly detector wins across heterogeneous time series, so automated solutions are needed. Existing ones face a trade-off:

<p align="center">
  <img src="assets/automatic_solutions.png" width="90%" alt="Automatic solutions for TSAD: model selection, ensembling, generation"/>
</p>

| Limitation | Affects | How RAMSAD addresses it |
|---|---|---|
| **L1** — must run the whole detector pool (expensive on long series / large pools) | Ensembling, model generation | Runs only the 1 (or *N*) selected detector(s) |
| **L2** — must be retrained when the detector pool changes | Trained selectors (classification / regression) | Non-parametric: just update the performance table |
| **L3** — generalizes poorly under domain shift | Trained selectors | Retrieval in a semantically aligned TSFM space |

> [!NOTE]
> "Training-free" refers to the **selector**. (i) Like any supervised selector, RAMSAD needs a labeled knowledge base to compute detector performance, but **no labels for the query**. (ii) The optional alignment module is trained offline, never sees detector identities or performance, and queries need **no description**. (iii) A frozen, off-the-shelf TSFM without alignment already outperforms all trained selectors.

## How it works

<p align="center">
  <img src="assets/overview.png" width="100%" alt="RAMSAD overview"/>
</p>

**(a) Similarity space creation.** Every series in the knowledge base is split into fixed-length segments of size `min(L, 8192)` (8192 = Chronos-2 context). Segments are embedded with a frozen TSFM (Chronos-2); each dataset's textual description is embedded with a frozen sentence transformer (`all-mpnet-base-v2`).

**(b) Alignment phase (optional).** A lightweight MLP projection head per modality is trained with a *symmetric multi-positive contrastive loss*, pulling each time-series segment towards all semantically compatible descriptions (description cosine similarity ≥ γ = 0.95, same-series pairs masked out). This injects a coarse domain prior into the time-series space, which mainly helps OOD. Only the **time-series head** is used afterwards.

**(c) Detector retrieval.** A query is segmented, embedded, and projected the same way. Retrieval is time-series-to-time-series:

<table>
<tr>
<td width="62%">

1. **Segment retrieval** — each of the *m* query segments retrieves its top-*k* nearest knowledge-base segments (cosine similarity).
2. **Segment-to-series aggregation** — for each candidate series $T_j$, average the distances of its retrieved segments, $\mathrm{base}(T_j)$, and measure how consistently it is retrieved, $\mathrm{freq}(T_j)$ = fraction of query segments that hit it. Rank by

   $$\mathrm{score}(T_j) = \frac{\mathrm{base}(T_j)}{\max(\epsilon, \mathrm{freq}(T_j))}$$

   and keep the top-*k* series. This rewards series that match *consistently and closely*, and suppresses isolated matches.
3. **Model selection** — look up the neighbors' rows in the precomputed performance matrix $\mathbf{V}$ (VUS-PR per series × detector), average per detector, and pick

   $$\widehat{D}(T_q) = \arg\max_{D} \frac{1}{k}\sum_{T \in \mathrm{top}_k(T_q)} \mathbf{V}(T, D)$$

   For an ensemble, take the *N* detectors with the highest averages.

</td>
<td width="38%">
<img src="assets/segment_to_series.png" width="100%" alt="Segment-to-series aggregation"/>
</td>
</tr>
</table>

Because the full performance profile is averaged, a detector is chosen only if it is broadly strong across the neighborhood — neighbors do not have to agree on their individual best detector:

<p align="center">
  <img src="assets/explanation_example.png" width="90%" alt="Example of a RAMSAD decision"/>
</p>

<details>
<summary><b>Semantic alignment details</b></summary>

<p align="center">
  <img src="assets/alignment.png" width="32%" alt="Before/after alignment"/>
  &nbsp;&nbsp;
  <img src="assets/alignment_arch.png" width="52%" alt="Alignment head architecture"/>
</p>

- Backbones (Chronos-2, `all-mpnet-base-v2`) are **frozen**; only the two MLP projection heads (hidden width = 2× projection dim, ℓ2-normalized outputs) are trained.
- Standard one-positive contrastive learning wrongly treats near-synonymous descriptions (e.g. two different server-monitoring datasets) as negatives. The **multi-positive** target matrix $P$ marks every description with cosine similarity ≥ 0.95 as a positive, and the loss is applied in both directions (segment→description and description→segment).
- Defaults: AdamW, lr 1e-5, weight decay 1e-4, 300 epochs, 10-epoch linear warmup + cosine decay, batch size 8192, temperature 0.1, grad-clip 1.0, 80/20 stratified train/val split.
- In OOD experiments the alignment heads are trained **without** the target domain and applied to it without retraining.

</details>

**Complexity.** With exact search, a batch of *b* query segments against *n* indexed segments of dimension *d* costs $\mathcal{O}(bnd)$; storage is $\mathcal{O}(nd + |\mathfrak{D}|M)$ for the embeddings and the performance matrix.

## Results

**Setup.** [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) univariate benchmark: 870 series from 9 domains, with the train/test split of recent model-selection benchmarks (619 train series → 3,264 knowledge-base segments; 251 test series → 1,383 query segments). Pool of **32 detectors**, compared against **25 automated solutions** (model selection, internal/unsupervised selection, ensembling, model generation, and reference baselines). Metric: **VUS-PR**. OOD = leave-one-domain-out over the 9 domains (target domain removed from the knowledge base *and* from alignment training).

<p align="center">
  <img src="assets/boxplots_cd.png" width="100%" alt="VUS-PR boxplots and critical difference diagrams"/>
</p>
<p align="center"><em>VUS-PR for in-distribution (a) and out-of-distribution (c), with critical-difference diagrams (b, d).</em></p>

| Setting | RAMSAD (N=1) | RAMSAD ens. (N=10) | Best trained selector (SATZilla) | Best full-pool ensemble (OE) |
|---|---|---|---|---|
| In-distribution | **0.74** | **0.77** | 0.68 | 0.64 |
| Out-of-distribution | 0.57 | **best** (+2% vs OE, ×4.7 faster) | 0.49 | 0.64 |

**Accuracy vs. runtime, and sensitivity to *k* and *N*.** RAMSAD is stable across *k* (max variation ≈0.04 in ID, ≤0.07 in OOD for N=1, shrinking as N grows) and always beats SATZilla.

<p align="center">
  <img src="assets/runtime_vs_accuracy.png" width="48%" alt="Runtime vs accuracy by k and N"/>
  &nbsp;
  <img src="assets/similarity_spaces.png" width="48%" alt="Similarity space comparison"/>
</p>
<p align="center"><em>Left: accuracy vs. detector runtime for different k and ensemble sizes N. Right: raw-space (Euclidean, DTW), TSFM (light) and aligned-TSFM (dark) retrieval spaces.</em></p>

**Similarity spaces.** The best raw-space distance (DTW) reaches 0.69 ID / 0.44 OOD. Plain Chronos-2 embeddings reach 0.75 ID / 0.59 OOD. Semantic alignment is roughly on par in ID and **improves OOD** — it is an OOD-oriented refinement, not a prerequisite.

**Selection latency** (median ms from raw query to recommendation):

| | **RAMSAD** | ARGOSMART | ISAC | MSAD | SATZilla | UReg | MetaOD | CFact |
|---|---|---|---|---|---|---|---|---|
| Features | **19.6** | 37.6 (catch22) | ← | ← | ← | ← | ← | ← |
| Inference | 0.05 | 1.1 | 25.6 | 25.8 | 39.7 | 221.8 | 426.7 | 453.4 |
| **Total** | **19.6** | 38.8 | 63.3 | 63.6 | 77.1 | 258.0 | 463.4 | 489.6 |

**Knowledge-base coverage and detector-pool size.** With only 10% of the knowledge base RAMSAD still beats SATZilla in OOD; in ID it is comparable from 20% and better from 80%. With half the detector pool it matches or beats SATZilla.

<p align="center">
  <img src="assets/kb_pool_sensitivity.png" width="60%" alt="Knowledge base and detector pool reduction"/>
</p>

---

## Quick start

> [!IMPORTANT]
> Run the full pipelines end-to-end via the provided bash scripts for ID and OOD:
>
> ```bash
> # In-distribution (ID) run (sets RAMSAD_DATA_ROOT=data/processed_data and defaults perf CSVs under data/raw/VUS/)
> bash scripts/run_full_pipeline_id_k1_k10.sh
>
> # Out-of-distribution (OOD) run (example domain: medical)
> RAMSAD_OOD_DOMAIN=medical bash scripts/run_full_pipeline_ood_k1_k10.sh
> ```
>
> Each driver creates `.venv`, builds segments, embeds descriptions and series, trains the alignment heads, retrieves, and runs model selection for `k = 1 … 10`. Results land in `outputs/full_pipeline_<split>_<stamp>/eval_k/k*/model_selection_k*.csv`.

### Reproducing the paper numbers

Expected mean VUS-PR (single detector, N = 1) from a clean run of the two drivers, against the paper's appendix tables. Small differences come from alignment training and GPU non-determinism.

| Setting | k = 1 | k = 3 | k = 6 (default) | k = 10 | Paper (k = 6) |
|---|---|---|---|---|---|
| ID, aligned Chronos-2 (`run_full_pipeline_id_k1_k10.sh`) | 0.714 | 0.731 | **0.742** | 0.720 | 0.738 |
| ID, plain Chronos-2 (`retrieve.mode=chronos`) | – | 0.717 | **0.744** | 0.713 | 0.746 |
| OOD, aligned Chronos-2, all 9 domains (`RAMSAD_OOD_DOMAIN=all`) | 0.517 | 0.568 | **0.580** | 0.595 | 0.582 |

Oracle: 0.866 (ID), 0.853 (OOD). On two Quadro RTX 6000 GPUs, the ID driver takes about 10 minutes and the full OOD sweep about 45 minutes, since it trains alignment once per held-out domain.

> [!NOTE]
> RAMSAD **selects** the ensemble (`select.n=N`), but scoring an ensemble requires running the *N* selected detectors on the raw series and combining their anomaly scores (e.g. with [TSB-AD](https://github.com/TheDatumOrg/TSB-AD)). The performance table only holds per-detector VUS-PR, so ensemble VUS-PR (the "RAMSAD (ens.)" results) cannot be computed from this repository alone.

OOD domains: `environment`, `facility`, `finance`, `humanactivity`, `medical`, `sensor`, `synthetic`, `traffic`, `webservice`. Use `RAMSAD_OOD_DOMAIN=all` to run every domain in turn and get the aggregated OOD score.

### ID vs. OOD: same pipeline, different files

The OOD experiment runs **exactly the same pipeline** as ID; only the input files change:

| | Knowledge base (train files) | Queries (test files) | Alignment trained on |
|---|---|---|---|
| **ID** | 619 TSB-AD train series (`data/raw/VUS/train.csv`) | 251 TSB-AD test series (`data/raw/VUS/test.csv`) | the knowledge base |
| **OOD** (domain *D*) | all 870 series **except** domain *D* (`ood/OOD_D/train_*`, `train_perf_OOD_D.csv`) | all series **of** domain *D* (`ood/OOD_D/test_*`, `test_perf_OOD_D.csv`) | the knowledge base (domain *D* never seen) |

For OOD, the ID train and test series are pooled and each domain is held out in turn (leave-one-domain-out), so across the 9 domains every one of the 870 series is queried once. The target domain is absent from the knowledge base, from the performance table, and from alignment training, and nothing is retrained for it. The OOD driver switches all of these files automatically; with Hydra, `data=ood data.ood_domain=<D>` does the same for embedding, retrieval, and selection.

## Install

```bash
git clone git@github.com:Hendrix8/RAMSAD.git && cd RAMSAD
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`[dev]` pulls in `sentence-transformers` for tests and for `ramsad-embed-desc`. For a minimal install use `pip install -e "."` and `pip install -e ".[desc]"` only when you need description embeddings.

Core embedding for time series uses `chronos-forecasting` and PyTorch (CUDA optional).

### PyTorch with NVIDIA GPU (CUDA)

CUDA builds are published on PyTorch's package index, not on PyPI alone. After [choosing a CUDA version](https://pytorch.org/get-started/locally/) that matches your driver, install ramsad like this (example: **CUDA 12.4**):

```bash
pip install -e ".[dev,cuda124]" --extra-index-url https://download.pytorch.org/whl/cu124
```

For another CUDA tag (`cu121`, `cu126`, …), use the same `--extra-index-url` URL PyTorch shows for that stack, and either keep `cuda124` (only re-resolves `torch`) or install `torch`/`torchvision` first with their command, then `pip install -e ".[dev]"`.

If PyTorch was already installed from PyPI, **reinstall** it with the extra index so the wheel is replaced:

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

### Optional extras

| Extra | Adds |
|---|---|
| *(base)* | Chronos + `torch`, Hydra, pandas, pyarrow |
| `[dev]` | pytest, ruff, `sentence-transformers` |
| `[desc]` | MPNet for `ramsad-embed-desc` |
| `[alignment]` | alignment training + viz + `sentence-transformers` |
| `[cuda124]` | re-resolves `torch` — use with `--extra-index-url` (see above) |

Example: `pip install -e ".[dev,alignment,cuda124]" --extra-index-url https://download.pytorch.org/whl/cu124`.

### Console entrypoints

With an activated venv: `ramsad-pipeline`, `ramsad-embed`, `ramsad-embed-desc`, `ramsad-retrieve`, `ramsad-select`, `ramsad-train-alignment`, `generate_segments` — same as `python -m ramsad.run_*` for RAMSAD stages and `python -m ramsad.generate_segments` for segment generation.

## Data

### Layout

| Path | Role |
|------|------|
| [`data/raw/VUS/train.csv`](data/raw/VUS/train.csv), [`data/raw/VUS/test.csv`](data/raw/VUS/test.csv) | Default **performance** tables (VUS-PR per model, `file` column + numeric model columns). Used by Hydra ID config (`src/ramsad/conf/data/id.yaml`) and by the full-pipeline scripts (`TRAIN_CSV` / `TEST_CSV`). Override with env vars when calling the scripts. |
| `data/raw/Train/`, `data/raw/Test/` | **Raw** univariate series CSVs (619 train / 251 test, from TSB-AD) referenced by the `file` column. Pass `--raw-root /path/to/data/raw` (parent of `Train/` and `Test/`) to segment generation. |
| [`data/raw/descriptions.csv`](data/raw/descriptions.csv) | Optional `source,description` metadata (one textual description per source dataset); full-pipeline scripts default `DESCRIPTIONS_CSV` here. |
| `data/processed_data/` | **Generated** segment and series parquets (`id/`, `ood/`, …). Default `RAMSAD_DATA_ROOT` for `scripts/run_full_pipeline_*`; listed in `.gitignore`. |

OOD splits pool the ID train and test series and hold out one domain at a time (see [ID vs. OOD](#id-vs-ood-same-pipeline-different-files)). OOD **Hydra** configs expect leave-domain-out artifacts under `${RAMSAD_DATA_ROOT}/ood/…` (see `src/ramsad/conf/data/ood.yaml`): segment parquets in `ood/OOD_<domain>/` and perf CSVs in `ood/Datasets/OOD/`.

### Environment variables

Paths resolve from the **current working directory** unless you use absolute paths or env overrides.

```bash
# Where segment/series parquets live (id/, ood/). Full-pipeline scripts export:
#   RAMSAD_DATA_ROOT=$REPO/data/processed_data
export RAMSAD_DATA_ROOT=/path/to/data/processed_data

# Hydra default if unset: ./data/splits (see src/ramsad/conf/paths/default.yaml).
# For this repo's layout, keep RAMSAD_DATA_ROOT pointing at processed_data so ramsad,
# alignment training, and retrieval agree.

export RAMSAD_OUTPUTS_ROOT=/path/to/outputs   # optional; default ./outputs
```

### Generate segment parquets

From performance CSVs and raw AD CSVs:

```bash
export RAMSAD_DATA_ROOT="$PWD/data/processed_data"

generate_segments --id \
  --train-csv data/raw/VUS/train.csv \
  --test-csv data/raw/VUS/test.csv \
  --data-root "$RAMSAD_DATA_ROOT" \
  --raw-root "$PWD/data/raw"
```

`--raw-root` must contain `Train/` and `Test/` folders whose files match the `file` column. Outputs include:

- `$RAMSAD_DATA_ROOT/id/train_segments.parquet`
- `$RAMSAD_DATA_ROOT/id/test_segments.parquet`
- `$RAMSAD_DATA_ROOT/id/train_series.parquet`
- `$RAMSAD_DATA_ROOT/id/test_series.parquet`

For OOD (refresh ID, then build leave-domain-out splits):

```bash
generate_segments --ood \
  --train-csv data/raw/VUS/train.csv \
  --test-csv data/raw/VUS/test.csv \
  --data-root "$RAMSAD_DATA_ROOT" \
  --raw-root "$PWD/data/raw" \
  --domains auto
```

If you omit `--descriptions-csv`, the tool looks for `<data-root>/descriptions.csv`; the bash drivers pass `data/raw/descriptions.csv` explicitly.

Use `--domains sensor,medical` to restrict OOD domains. Existing outputs are skipped unless `--force` is passed.

Standalone OOD layout from ID parquets + perf CSVs only:

```bash
python scripts/build_ood_splits_from_id.py \
  --id-root "$RAMSAD_DATA_ROOT/id" \
  --out-root "$RAMSAD_DATA_ROOT/ood" \
  --train-csv data/raw/VUS/train.csv \
  --test-csv data/raw/VUS/test.csv
```

<details>
<summary><b>OOD segment / series parquets (reference)</b></summary>

Hydra expects the following files for each domain `DOMAIN` (e.g. `sensor`), under **`${RAMSAD_DATA_ROOT}/ood/`**:

- `OOD_${DOMAIN}/train_segments_OOD_${DOMAIN}.parquet`
- `OOD_${DOMAIN}/test_segments_OOD_${DOMAIN}.parquet`
- `OOD_${DOMAIN}/train_series_OOD_${DOMAIN}.parquet`
- `OOD_${DOMAIN}/test_series_OOD_${DOMAIN}.parquet`
- `Datasets/OOD/train_perf_OOD_${DOMAIN}.csv`, `test_perf_OOD_${DOMAIN}.csv`

**Generate from ID:** use `generate_segments --ood` or [`scripts/build_ood_splits_from_id.py`](scripts/build_ood_splits_from_id.py) (see above).

Regenerate `*_series_*.parquet` from segments if needed:

```bash
python scripts/build_series_parquets_from_segments.py \
  --train-segments "$RAMSAD_DATA_ROOT/ood/OOD_sensor/train_segments_OOD_sensor.parquet" \
  --test-segments "$RAMSAD_DATA_ROOT/ood/OOD_sensor/test_segments_OOD_sensor.parquet" \
  --out-train-series "$RAMSAD_DATA_ROOT/ood/OOD_sensor/train_series_OOD_sensor.parquet" \
  --out-test-series "$RAMSAD_DATA_ROOT/ood/OOD_sensor/test_series_OOD_sensor.parquet"
```

</details>

## Run (Hydra)

From your **project directory** (any cwd you choose; relative paths resolve from there). Set `RAMSAD_DATA_ROOT` first so data paths match [`src/ramsad/conf/data/id.yaml`](src/ramsad/conf/data/id.yaml) (perf CSVs under `data/raw/VUS/` relative to the repo).

**Full pipeline** (embed → retrieve → select), OOD example:

```bash
python -m ramsad.run_pipeline experiment=paper_ood_single
```

ID example:

```bash
python -m ramsad.run_pipeline experiment=paper_id_single
```

**Stages only:**

```bash
python -m ramsad.run_embed        # Chronos-2 segment embeddings
python -m ramsad.run_embed_desc   # MPNet description embeddings
python -m ramsad.run_retrieve     # segment retrieval + segment-to-series aggregation
python -m ramsad.run_select       # mean-VUS argmax over the top-k neighbors
```

Or: `ramsad-embed`, `ramsad-embed-desc`, … if the venv is on `PATH`.

### Useful overrides

- OOD domain: `data.ood_domain=medical`
- Skip Chronos embed if parquets already have `amazon_chronos-2`: `pipeline.skip_embed=true`
- **Description embeddings (MPNet)** on train/test segment parquets: `pipeline.embed_desc=true` or run `python -m ramsad.run_embed_desc` (defaults: `embed_desc.text_column=desc`, `embed_desc.out_column=all-mpnet-base-v2_desc`; use `data=id` / `data=ood` like the main pipeline). Install: `pip install -e ".[desc]"` (or `[dev]`).
- Neighbor count for selection: `select.k=6` (default, as in the paper; must be ≤ `retrieve.top_k`, default 10).
- Number of selected detectors: `select.n=1` (default). With `select.n=N>1`, the output's `selected_models` column lists the top-*N* detectors by neighbor mean VUS (`;`-separated), which you then run and combine as an ensemble. `selected_model` is always the top-1 detector. To reuse one retrieval CSV for many `k`, set `select.retrieval_csv=/abs/path/to/retrieval_topk.csv` (see `scripts/run_full_pipeline_id_k1_k10.sh` and `scripts/run_full_pipeline_ood_k1_k10.sh`).
- **Alignment retrieval**: see [Alignment retrieval paths](#alignment-retrieval-paths) below.

Outputs go to Hydra's run directory under **`paths.outputs_root`** (default `./outputs`; see console log for the resolved `output_dir`).

Configs under `src/ramsad/conf/experiment/` include `noop` (default), `paper_ood_single`, `paper_id_single`, and `paper_ood_full` (comments show a bash loop for all nine OOD domains).

### One-shot driver (`k = 1 … 10`)

The ID and OOD drivers create `.venv`, install `.[dev,alignment]` (optional **`INSTALL_CUDA124=1`** for the PyTorch CUDA index), run **segment generation → desc embeddings → Chronos → alignment training → aligned retrieval** (`top_k=10`), then **model selection** for **`k = 1` through `10`** into `outputs/full_pipeline_<split>_<stamp>/eval_k/k*/model_selection_k*.csv` (same retrieval CSV for all `k`). Segment generation is skipped automatically when the required parquets already exist; set `FORCE_SEGMENTS=1` to rebuild. See the script headers for **`TRAIN_CSV`**, **`TEST_CSV`**, **`RAW_ROOT`**, **`DESCRIPTIONS_CSV`**, **`SKIP_*`**, and **`RET_CSV_OVERRIDE`**.

```bash
bash scripts/run_full_pipeline_id_k1_k10.sh
RAMSAD_OOD_DOMAIN=medical bash scripts/run_full_pipeline_ood_k1_k10.sh
INSTALL_CUDA124=1 bash scripts/run_full_pipeline_id_k1_k10.sh
```

For a fresh data build from raw CSVs:

```bash
RAW_ROOT=/path/to/data/raw FORCE_SEGMENTS=1 bash scripts/run_full_pipeline_id_k1_k10.sh
RAW_ROOT=/path/to/data/raw FORCE_SEGMENTS=1 RAMSAD_OOD_DOMAIN=sensor bash scripts/run_full_pipeline_ood_k1_k10.sh
```

### K-sweep evaluation

After selection, `scripts/evaluate_k_sweep.py` and `scripts/evaluate_all_ood_k_sweep.py` compare chosen models to oracle VUS on the test perf CSV. The summary CSV columns include **`N_series`** (number of test series scored, i.e. matched rows—not "number of benchmark datasets").

## Description embeddings (`all-mpnet-base-v2_desc`)

Segment parquets need a text column (default **`desc`**) and store description-side vectors in a column such as **`all-mpnet-base-v2_desc`** (768-d lists, same style as Chronos columns). Pre-built segments may only have **`all-mpnet-base-v2_class`** until you run this step.

**Standalone** (uses `conf/embed_desc.yaml`, default `data=id`):

```bash
python -m ramsad.run_embed_desc
python -m ramsad.run_embed_desc data=ood data.ood_domain=sensor
```

**Inside the full pipeline** (before Chronos embed):

```bash
python -m ramsad.run_pipeline experiment=paper_id_single pipeline.embed_desc=true
```

**Overrides:** e.g. `embed_desc.text_column=Description embed_desc.out_column=all-mpnet-base-v2_desc embed_desc.device=cpu`

After writing **`all-mpnet-base-v2_desc`**, point alignment training at it: `ramsad-train-alignment data.desc_emb_col=all-mpnet-base-v2_desc`.

## Alignment retrieval paths

`retrieve.mode=alignment` loads **`checkpoints/best_model.pt`** and **`config.yaml`**.

**Defaults** (relative to **cwd**; same roots as `paths.outputs_root`, default `./outputs` or `RAMSAD_OUTPUTS_ROOT`):

- `retrieve.alignment_run_dir=${paths.outputs_root}/alignment` — must match where alignment training writes (see `alignment/config/config.yaml` `output_dir`).
- `retrieve.alignment_emb_column=emb_aligned` — in-memory column name for projected vectors.

If `alignment_run_dir` points at that **base** folder and there is **no** `best_model.pt` directly inside it, retrieval **automatically picks the newest** immediate subfolder (e.g. `run-20260430-021630-infonce`) that contains `checkpoints/best_model.pt`. You can still override with the exact inner `run-...` path.

1. **Train** (see next section). The log prints **`Output directory:`** — that folder always works if passed explicitly. Do **not** pass the ramsad **pipeline** Hydra folder (e.g. `outputs/2026-04-30/02-16-29` from `run_pipeline`); that job does not contain an alignment checkpoint.

2. **Minimal alignment run** (defaults):

   ```bash
   python -m ramsad.run_pipeline experiment=paper_id_single retrieve.mode=alignment
   ```

3. **Overrides:** set `retrieve.alignment_run_dir=...` to another base or a specific `run-...` folder; set `retrieve.alignment_emb_column=...` if you want a different column name.

4. **`retrieve.alignment_source_column`:** Chronos column to project; default `auto` uses `embed.embed_column` (e.g. `amazon_chronos-2`).

5. **GPU:** If PyTorch warns about an old NVIDIA driver, projection may use **CPU**; set `retrieve.alignment_device=cpu` to force CPU.

## Train embedding alignment (optional)

The vendored trainer lives under `alignment/` (Hydra). From your **project directory**, install extra deps and run:

```bash
pip install -e ".[dev,alignment]"
ramsad-train-alignment   # same as: python alignment/train.py
```

[`alignment/config/data/default.yaml`](alignment/config/data/default.yaml) defaults use **`${RAMSAD_DATA_ROOT:-data/processed_data}/id/train_segments.parquet`** for segments and **`data/raw/VUS/train.csv`** for `performance_file` (Series-to-Performance alignment). Bundled or freshly built segments may include **`all-mpnet-base-v2_class`** only until you run [description embedding](#description-embeddings-all-mpnet-base-v2_desc); then use `data.desc_emb_col=all-mpnet-base-v2_desc`. Training writes under `./outputs/alignment/run-...` by default, or `$RAMSAD_OUTPUTS_ROOT/alignment/run-...` if set (see **`Output directory:`** in the log). Retrieval uses the same base (`paths.outputs_root/alignment`) and selects the **newest** run unless you override `retrieve.alignment_run_dir`.

Contrastive alignment between time-series embeddings (e.g. Chronos) and description embeddings. Layout:

```
alignment/
├── config/           # Hydra: config.yaml, data/, model/, training/
├── data/dataset.py
├── model/embedding_alignment.py
├── trainer/trainer.py
├── utils/
└── train.py          # Main entry (also: ramsad-train-alignment)
```

Override examples:

```bash
python alignment/train.py training.batch_size=256 data.desc_emb_col=all-mpnet-base-v2_desc
python alignment/train.py model=large training=fast
```

Checkpoints and logs go under `${RAMSAD_OUTPUTS_ROOT:-./outputs}/alignment/run-*` (exact path printed as **`Output directory:`**).

## Tests

```bash
pytest -q
```

(Embedding is not exercised in CI tests; retrieve/select use small synthetic parquets.)

## Repository layout

```
.
├── src/ramsad/          # library + run_*.py Hydra entrypoints
│   ├── embed.py             # Chronos-2 segment embeddings
│   ├── embed_desc.py        # MPNet description embeddings
│   ├── retrieve.py          # segment retrieval + segment-to-series aggregation
│   ├── select.py            # mean-VUS argmax model selection
│   ├── alignment_projector.py
│   ├── generate_segments.py
│   └── conf/                # config.yaml, data/, embed/, retrieve/, select/, experiment/, …
├── alignment/           # Chronos↔description alignment trainer (train.py, model/, trainer/, …)
├── scripts/             # run_full_pipeline_*, evaluate_k_sweep.py, evaluate_all_ood_k_sweep.py,
│                        # build_ood_splits_from_id.py, build_series_parquets_from_segments.py
├── data/raw/            # performance CSVs (VUS/), raw series (Train/, Test/), descriptions.csv
├── data/processed_data/ # generated parquets (gitignored)
├── tests/
└── assets/              # README figures (from the paper)
```

Raw inputs under **`data/raw/`** are tracked; **`data/processed_data/`** holds regenerated artifacts and is **gitignored**. Runtime dirs such as `.venv/`, `outputs/`, `multirun/`, `__pycache__/`, and `.pytest_cache/` should not be committed.

## Acknowledgements

Supported by DataGEMS (101188416). This work was granted access to the HPC resources of IDRIS under the allocation 2025-A0191012641 made by GENCI. The benchmark data and detector pool come from [TSB-AD](https://github.com/TheDatumOrg/TSB-AD); time-series embeddings use [Chronos-2](https://github.com/amazon-science/chronos-forecasting).

## License

[MIT](LICENSE)
