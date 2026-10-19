# Target-Relative Structural Features for Class Comment Classification

**CONFIRMED — scope.** A standalone reproducibility project for multi-label classification of Python and Pharo class-comment targets. It reconstructs enclosing comments from pinned NLBSE releases, locates unambiguous target spans, and evaluates four structural features, frozen MiniLM, their combination, matched feature scaling, and TF-IDF controls. Supervised SetFit is an optional, more expensive reference.

**RESULT — supported claim.** On the recovered Python/Pharo cohorts, four structural features improve grouped Macro-F1 over frozen MiniLM by 0.1242/0.1049. Structure alone is a strong reference; concatenation is not uniformly better than structure alone. These are local Train-derived evaluations, not official competition Test scores.

## Setup

The verified workflow uses **two environments**. Run commands from the repository root. Install Python **3.13.9** for core fitting/reporting and **3.10.19** for encoding/SetFit. The commands below define `CORE` and `ENCODER` for the remaining examples.

Windows PowerShell:

```powershell
py -3.13 -m venv env-core
py -3.10 -m venv env-encoder
function CORE { & .\env-core\Scripts\python.exe @args }
function ENCODER { & .\env-encoder\Scripts\python.exe @args }
CORE -m pip install -r locks/core.txt
ENCODER -m pip install -r locks/encoder.txt
```

POSIX shell (installation and numerical equivalence have not been verified):

```sh
python3.13 -m venv env-core
python3.10 -m venv env-encoder
CORE() { env-core/bin/python "$@"; }
ENCODER() { env-encoder/bin/python "$@"; }
CORE -m pip install -r locks/core.txt
ENCODER -m pip install -r locks/encoder.txt
```

The Python selectors above select the minor version; confirm the patch versions with `CORE --version` and `ENCODER --version`. `locks/` pins the full recorded dependency closure; the shorter `requirements-core.txt` and `requirements-encoder.txt` pin the direct dependencies. CUDA verification used PyTorch 2.2.0+cu118. CPU encoding is supported by `--device cpu`, but bitwise CPU equivalence has not been established. The supplied encoder pins describe the recorded Windows/CUDA environment; other platforms may require different PyTorch packages, which would be an unverified environment.

## Data preparation

**CONFIRMED:** no datasets, comment text, embeddings, trained models, or checkpoints are distributed. See [DATA_PERMISSIONS.md](DATA_PERMISSIONS.md). Use only these four files:

- `python_original.csv`, `pharo_original.csv`: archived [NLBSE 2024 `project_classes` files](https://github.com/nlbse2024/code-comment-classification/tree/0a1d728e7058bd572c35e97a88efff1f5b92b1ae).
- `python_train.parquet`, `pharo_train.parquet`: [NLBSE 2026 Train splits](https://huggingface.co/datasets/NLBSE/nlbse26-code-comment-classification/tree/d0dc51c11dbb660ba08f60bd714f4733e9560f3b).

Pinned URLs, byte sizes and SHA-256 hashes are in `configs/data_sources.json`. Supply legally obtained files in `../local-data`, or explicitly acquire only those files:

```text
CORE acquire.py data --destination ../local-data
```

Acquire the pinned encoder separately (91.6 MB download), after accepting its model terms and resource cost:

```text
ENCODER acquire.py checkpoint --destination ../local-checkpoint
```

The checkpoint is `sentence-transformers/paraphrase-MiniLM-L6-v2` at revision `c9a2bfebc254878aee8c3aca9e6844d5bbb102d1`. Every checkpoint file is hash-checked. No model training is part of the main frozen workflow. The pipeline is offline after acquisition and never invokes a dataset split loader. Do not supply official Test files.

## Quick start

Run the synthetic smoke test without data or a checkpoint:

```text
CORE -m unittest discover -s tests -v
```

## Full core reproduction

Choose a fresh work directory **outside this source repository**. The preparation and fitting stages refuse to overwrite an existing run.

```text
CORE run.py prepare --input-dir ../local-data --work-dir ../run-core
ENCODER run.py embed --work-dir ../run-core --checkpoint-dir ../local-checkpoint --device cuda
CORE run.py fit --work-dir ../run-core
CORE run.py report --work-dir ../run-core
CORE run.py check --work-dir ../run-core
```

The fixed equations and fitting blocks preserve the verified implementation; `configs/source_provenance.json` records source symbols and hashes. Input files and the local checkpoint are hash-checked. TF-IDF arms are fitted locally using the fixed configuration.

**CONFIRMED — input and features.** Frozen semantics encode the target's first 128 whitespace tokens followed by the source class identifier's first 16 tokens. This is a single class-identifier input, not a candidate-category-conditioned input. MiniLM uses its default 128 WordPiece cap, 384 dimensions, normalized float32 embeddings, batch 32, evaluation mode and no gradients. Given the original comment `O` and raw target span `[a,b)`, structural features are:

1. `a / max(1,len(O))`;
2. `(b-a) / max(1,len(O))`;
3. `log1p(len(O[:a].split()))`;
4. `log1p(len(O[b:].split()))`.

Character offsets index Python Unicode strings, not bytes. Recovery tries raw, space-normalized and deletion-normalized exact matches in that order, requires a unique source and unique span, and excludes targets lacking external word context. Lossy legacy matching is diagnostic only. Duplicate source records collapse. Linked normalized target/source texts form dependence components. Three grouped folds balance component row counts deterministically; seven project holdouts purge linked training components. Original duplicate/conflicting annotation rows are retained.

## Evaluation protocols

**CONFIRMED:** both protocols use only the recovered context-bearing Train cohorts, with five fixed Python labels and six fixed Pharo labels:

- `group`: three deterministic folds that keep linked original-comment components together.
- `project`: seven leave-one-project-out splits per language, purging training components linked to the held-out project.

Scores pool out-of-fold predictions within each language/protocol. Together the protocols generate 20 train/evaluation partitions. They are distinct from the upstream official Test protocol.

All core binary heads use `C=1`, balanced weights, liblinear, `max_iter=1000`, `tol=1e-4`, seed **20261018**, and threshold `p>=0.5`; a single-valued training label uses its constant probability. Structural scaling is fitted on training rows only. `SP_matched` divides standardized structure by its training RMS row norm before concatenation, without renormalizing the whole row. TF-IDF uses word 1–2 grams, 20,000 features, `min_df=1`, sublinear TF and L2 normalization. `configs/protocol.json` and source define the fixed settings; they are not a tuning interface.

## Expected outputs

**RESULT — pooled Macro-F1**, rounded to four decimals:

| Arm | Python group | Python project | Pharo group | Pharo project |
|---|---:|---:|---:|---:|
| `S`: frozen MiniLM | .3456 | .3314 | .3756 | .3343 |
| `P`: four structural features | .4976 | .4754 | .4624 | .4702 |
| `SP`: concatenation | .4698 | .4633 | .4805 | .4716 |
| `SP_matched`: RMS-matched structure | .4682 | .4504 | .4797 | .4759 |
| `T`: TF-IDF | .2772 | .2456 | .4072 | .3484 |
| `T_exact`: TF-IDF + structure | .4664 | .4467 | .4902 | .4927 |

Preparation recovers 1,091/803 uniquely localized Python/Pharo Train rows; the context-bearing evaluation cohorts contain **1,031/752 targets in 174/183 comments**. All 20 split memberships are generated under `cohort/splits.json`.

`models/metrics.json` contains pooled Macro/Micro-F1 and confidence intervals; `per_label.json` preserves every fixed label; `fold_project_consistency.json` preserves all folds/projects; `paired_contrasts.json` contains paired original-comment intervals and linked-component sensitivity. `T_coarse`, `T_nospan`, and `T_permuted` reproduce the fixed lexical ablations. Bootstrap uses 2,000 seed-20261018 percentile draws, fixed predictions and no refits. All labels enter Macro-F1, including zero-support labels with F1=0.

`expected/` holds full-precision numerical references, reconstruction/split hashes, figure 2 gains and figure 3 confusion counts/F1. `check` verifies reconstruction hashes and compares numerical results using absolute tolerance **1e-12**, zero relative tolerance, separately reporting exact matches. It regenerates figure data JSON outside the source tree. The framework figure has no empirical numerical series; figure graphics are not included.

## Optional supervised references

```text
ENCODER run.py setfit --input-dir ../local-data --work-dir ../run-core --checkpoint-dir ../local-checkpoint
ENCODER run.py setfit-report --work-dir ../run-core
```

This trains **20 fresh encoders** on the same training complements. It preserves the existing official-style schedule: seed 42, 10 epochs, batch 32, 20 pair iterations, exact released `combo` input, raw embeddings and default sklearn 1.4.0 `MultiOutputClassifier(LogisticRegression())` heads. `SetFit_P` appends training-standardized structure. The report also reproduces the fixed balanced, normalized, RMS-matched head control from those split-specific embeddings; it does not retrain the encoders again.

Expected official-style SetFit/SetFit+P grouped Macro-F1: Python **.3962/.3970**, Pharo **.5342/.5282**. Full expected Micro-F1, label results and paired intervals are included. Supervised `combo` differs from the frozen input described above; these are distinct reference families.

## Runtime and boundaries

**RESULT:** the original frozen encoding took 2.01 s for 1,782 unique inputs on an RTX 4060 Laptop, about 177 MiB peak CUDA allocation. Fixed core fitting took seconds; preparation and bootstrap add seconds. Library startup, dependency installation and checkpoint download are additional. An 8 GB GPU is sufficient for the recorded setup. Allow several GB of system memory for dependencies and bootstrap arrays. Optional SetFit took **9,508 s (2.64 h)** summed training time and about **1.41 GiB** peak CUDA allocation, with 20 saved encoders requiring about 2 GB disk.

**RESULT — release verification:** core cohorts, split membership, embeddings, probabilities, metrics, paired intervals and figure data were compared against saved evidence. All core numerical checks matched exactly on the recorded environment. Optional saved SetFit outputs and fixed head controls were verified; **full supervised encoder retraining was not repeated for this release**. A clean dependency installation and other operating systems/CPU inference were not validated.

**CONFIRMED — scientific boundaries:** these are previously examined, recovery-selected Train cohorts. Archived full comments may contain text overlapping separately released Test; official Test files are neither loaded nor compared. Results do not estimate all-Train/Test or naturally segmented deployment performance. Annotation-conditioned target boundaries, templates and recovery coverage limit generalization. Bootstrap intervals condition on fitted predictions, not encoder-training randomness or a new project population. The repository does not establish a new architecture or guarantee improved developer productivity.

## Acknowledgments and code license

Data: [NLBSE 2024 release](https://github.com/nlbse2024/code-comment-classification/tree/0a1d728e7058bd572c35e97a88efff1f5b92b1ae), [NLBSE 2026 release](https://huggingface.co/datasets/NLBSE/nlbse26-code-comment-classification/tree/d0dc51c11dbb660ba08f60bd714f4733e9560f3b), and the originating class-comment work by Rani et al. (JSS 181, 111047, 2021). Model: [Sentence Transformers paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2). The optional schedule follows the [pinned official SetFit baseline](https://github.com/nlbse2026/code-comment-classification/blob/65d7ef513a1d66bd0dddd35fa9bd2e273eacde92/SetFit_baseline.ipynb).

**CONFIRMED:** original repository code is licensed under the [MIT License](LICENSE), copyright (c) 2026 zuhak9793-pixel. Data, pretrained models, dependencies and third-party source comments retain their own terms; see [DATA_PERMISSIONS.md](DATA_PERMISSIONS.md).
