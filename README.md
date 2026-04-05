# Admissions Essay Assessment

A suite of NLP notebooks for analyzing admissions essay corpora — measuring stylometric features,
discourse structure, narrative arcs, topic diversity, and AI-generation risk.

The goal is to give reviewers interpretable, quantitative signals alongside qualitative reads,
not to automate decisions. Every output is designed to surface essays worth a closer look, with
enough diagnostic detail to understand why a flag was raised.

## Structure

```
essay_assessment/          # Core analysis notebooks
  stylometrics.ipynb       # Lexical richness, syntactic features, style clustering
  discourse_analysis.ipynb # Argumentative role labeling, coherence scoring, narrative arcs
  topic_modeling.ipynb     # BERTopic topic modeling across corpora
  diversity_analysis.ipynb # Pairwise semantic similarity, near-duplicate detection
  finetune_discourse_deberta.ipynb  # Fine-tuning DeBERTa for discourse role classification

ai_detection/              # AI-generation detection pipeline
  ai_detection.ipynb       # Four-test ensemble: semantic similarity, perplexity/burstiness,
                           # stylometric flatness, discourse coherence

data/
  synthetic/               # LLM-generated essays (committed)
  freeform/                # Freeform LLM essays (committed)
  real/                    # Real applicant essays (gitignored — see data/real/README.md)

outputs/                   # All computed CSVs and model outputs (gitignored — regenerate locally)

environment.yaml           # Conda environment definition
```

## Notebooks

### `stylometrics.ipynb`
Computes per-essay stylometric features across six sections: lexical richness (TTR, MTLD,
rare-word rate), syntactic complexity (dependency depth, clause density), surface and length
features, discourse marker density, sentence-length variation, and cross-corpus style clustering.
Outputs `stylometric_features.csv`, which feeds directly into the AI detection pipeline.

### `discourse_analysis.ipynb`
Applies the fine-tuned discourse classifier (see below) to label every sentence with an
argumentative role (Lead, Position, Claim, Evidence, etc.), then computes narrative arc
trajectories, argument density, and inter-sentence coherence scores. Run once with
`DATASET = 'real'` and once with `DATASET = 'synthetic'` to produce the comparison features
used in AI detection.

### `topic_modeling.ipynb`
Fits a BERTopic model across all corpora to discover latent topic clusters and compare their
distribution between real and synthetic essays. Includes an optional LLM-labeling step for
human-readable topic names that runs fully locally.

### `diversity_analysis.ipynb`
Embeds all essays and computes pairwise cosine similarities to measure semantic redundancy
within and across corpora. Designed in part to characterize the clustering behavior found in
the synthetic data — the same property that makes cosine similarity an effective AI detection
signal.

### `finetune_discourse_deberta.ipynb`
Fine-tunes `roberta-base` on the
[PERSUADE 2.0 dataset](https://www.kaggle.com/competitions/feedback-prize-2021) to classify
essay sentences into seven argumentative roles. Includes essay-level train/val/test splits,
focal loss for class imbalance, and evaluation metrics. Model weights are not committed due
to size — run locally or load a compatible checkpoint from HuggingFace Hub.

### `ai_detection/ai_detection.ipynb`
Four-test ensemble producing a combined risk score per essay. See `ai_detection/README.md`
for full details on weights, prerequisites, and outputs.

## Recommended Run Order

Some notebooks depend on outputs from others. Run in this order:

1. `finetune_discourse_deberta.ipynb` — produces the discourse classifier weights
2. `stylometrics.ipynb` — produces `stylometric_features.csv`
3. `discourse_analysis.ipynb` (twice: `DATASET = 'real'`, then `DATASET = 'synthetic'`)
4. `diversity_analysis.ipynb` — produces embedding-space similarity metrics
5. `topic_modeling.ipynb` — standalone, no upstream dependencies
6. `ai_detection/ai_detection.ipynb` — requires outputs from steps 2 and 3

## Setup

```bash
conda env create -f environment.yaml
conda activate essay-assessment-eda
python -m spacy download en_core_web_sm
```

PyTorch is installed with CUDA 12.1 support via the `pytorch` channel. If you're on CPU only,
remove the `pytorch-cuda` line from `environment.yaml` before creating the environment.

## Data

Synthetic and freeform essays were generated using GPT-5.4 / Claude Sonnet 4.6 and Opus 4.6
across a range of controlled dimensions:

- **Personas** — varied first-person narrator backgrounds and voices
- **Topics** — common Common App prompts and supplemental essay types
- **Registers** — formal, conversational, and mixed rhetorical styles
- **Replications** — multiple draws per condition to capture model variance

The synthetic corpus (`data/synthetic/`, 900 essays) includes a `proxy_score` column
(simulated holistic rating) and a `model` column for source attribution. The freeform corpus
(`data/freeform/`, 600 essays) uses a looser prompt structure without topic constraints or proxy scores.

### Using your own essays

Place your essay file at `data/real/essays.xlsx`. This directory is gitignored and never
distributed. The notebooks expect two columns at minimum:

| Column | Description |
|---|---|
| `Campus ID` | Unique student identifier |
| `Essay Response` | Full essay text |

If your export uses different column names, the loader will attempt to match against common
aliases before raising an error:

- **ID column:** `student_id`, `Campus ID`, `id`, `ID`, `student id`, `StudentID`
- **Essay column:** `essay`, `Essay Response`, `essay_text`, `text`, `Essay`, `response`

An optional `Essay Score` column (numeric holistic rating) is used in some stylometric
analyses but is not required to run the pipeline.

## Notes

- All notebooks write outputs to `outputs/` (gitignored to protect proprietary data).
- Notebooks use paths relative to the repo root (`../data/`, `../outputs/`), so run them
  from their respective directories or set the kernel working directory accordingly.
