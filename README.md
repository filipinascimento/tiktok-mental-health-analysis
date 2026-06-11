# TikTok Mental Health Analysis

Code for the data collection and analysis pipeline for:

**The Tone of Awareness: Topic, Sentiment, and Toxicity Maps During Mental Health Month on TikTok**

Authors:
Henrique Ferraz de Arruda, Andreia Sofia Teixeira, Pranay Gundala Reddy, Anindya Mondal, Kleber Andrade Oliveira, and Filipi Nascimento Silva.

Affiliations:

- Institute for Biocomputation and Physics of Complex Systems (BIFI), University of Zaragoza, Zaragoza, Spain
- ARAID Foundation, Zaragoza, Spain
- BRAN Lab, Network Science Institute, Northeastern University London, London, UK
- Kent Medway Medical School, Canterbury, United Kingdom
- LASIGE, Faculdade de Ciencias da Universidade de Lisboa, Lisboa, Portugal
- Observatory on Social Media, Indiana University, Bloomington, IN, USA
- Social Dynamics Research Lab, Department of Psychology, University of Limerick, Limerick, Ireland
- CSSI - Kellogg School of Management, Northwestern University, IL, USA

## Scope

This repository contains scripts and metadata only. It intentionally excludes raw TikTok data, intermediate data, model artifacts, generated figures, logs, and paper-rendered assets. TikTok data are governed by the TikTok Research API terms and cannot be redistributed here.

The pipeline covers:

- TikTok Research API video collection
- video-ID preparation for comment collection
- restartable comment collection
- preprocessing
- XLM-T sentiment, VADER sentiment, and Detoxify toxicity scoring
- sentence-transformer embeddings and BERTopic topic modeling
- log-odds topic keywords
- paper figures, tables, tests, and methodology audit
- Zenodo-shareable indexed metrics CSV export

## TikTok API Access

You need TikTok Research API access to collect new data. Set these environment variables before running collection scripts:

```bash
export TIKTOK_CLIENT_KEY="..."
export TIKTOK_CLIENT_SECRET="..."
```

Without these credentials, API collection stages cannot run. You can still run downstream stages only if you already have local data files in `Data/`.

## Repository Layout

```text
Scripts/          Pipeline entry points and analysis scripts
tiktokresearch/   Local package with the TikTokAPI wrapper and shared pipeline utilities
Data/             Generated/local data, ignored by Git
Figures/          Generated plots, ignored by Git
```

The local package exposes the API wrapper as:

```python
import tiktokresearch as tiktok

api = tiktok.TikTokAPI(client_key, client_secret)
```

## Installation

Use a GPU-enabled Python environment for the scoring and embedding stages when possible.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If you use Conda, create an environment with Python 3.10+ and install from `requirements.txt` or `pip install -e .`.


## Pipeline

Run from the repository root.

Dry-run video collection:

```bash
python Scripts/01_collect_videos.py --dataset mentalhealth2024 --dry-run
```

Collect videos using the paper keyword window:

```bash
python Scripts/01_collect_videos.py --dataset mentalhealth2024
```

Prepare IDs for comment collection:

```bash
python Scripts/02_prepare_comment_ids.py --dataset mentalhealth2024 --shuffle
```

Collect comments:

```bash
python Scripts/03_collect_comments.py --dataset mentalhealth2024
```

Preprocess:

```bash
python Scripts/04_preprocess.py --dataset mentalhealth2024
```

Score sentiment and toxicity:

```bash
python Scripts/05_score_sentiment_toxicity.py --dataset mentalhealth2024 --reuse
```

Fit/export embeddings and topics:

```bash
python Scripts/06_embed_and_topic_model.py --dataset mentalhealth2024 --reuse-embeddings --reuse-model
```

Generate paper figures, tables, tests, and audit:

```bash
python Scripts/07_generate_paper_outputs.py --dataset mentalhealth2024 --skip-umap
```

Export the shareable indexed metrics dataset for Zenodo:

```bash
python Scripts/08_export_zenodo_indexed_metrics_dataset.py \
  --variant-dir old/Variants/paper_locked_multilingual_detoxify_mpnet \
  --output-dir zenodo/indexed_metrics_dataset \
  --min-entries-per-date 10
```

This export writes only `videos.csv` and `comments.csv`: release-local video/comment indices, video topic labels, UTC dates, Detoxify multilingual toxicity scores, and XLM-T sentiment scores. It excludes TikTok IDs, usernames, raw text, processed text, captions, hashtags, URLs, and audio transcripts. Dates with fewer than 10 combined video/comment rows are excluded iteratively, and comments are retained only when their parent `video_index` remains in `videos.csv`.

The full wrapper is:

```bash
Scripts/run_full_pipeline.sh --dataset mentalhealth2024 --skip-umap
```

To include collection:

```bash
Scripts/run_full_pipeline.sh --dataset mentalhealth2024 --collect-videos --collect-comments --skip-umap
```

## Paper Defaults

The default collection terms are defined in `tiktokresearch.pipeline.MENTAL_HEALTH_TERMS` and cover the Mental Health Awareness Month window from April 15 through June 15 for 2023 and 2024.

The topic modeling defaults follow the manuscript settings:

- embeddings: `sentence-transformers/all-mpnet-base-v2`
- BERTopic with UMAP and HDBSCAN
- UMAP neighbors: `20`
- UMAP components: `10`
- UMAP min distance: `0.05`
- HDBSCAN min cluster size: `300`
- HDBSCAN max cluster size: `5000`
- HDBSCAN min samples: `15`
