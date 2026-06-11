from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiktokresearch import (
    DATASET_DEFAULT,
    coerce_identifier_columns,
    log_odds_keywords,
    model_device,
    repo_root_from_script,
    topic_preprocess_text,
    write_csv,
    write_feather,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute video embeddings and fit/export BERTopic topics.")
    parser.add_argument("--dataset", default=DATASET_DEFAULT)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--data-dir", type=Path, default=None, help="Defaults to <repo-root>/Data")
    parser.add_argument("--input", type=Path, default=None, help="Defaults to videos_enriched_<dataset>.feather")
    parser.add_argument("--docs-output", type=Path, default=None)
    parser.add_argument("--embedding-output", type=Path, default=None)
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--assignments-output", type=Path, default=None)
    parser.add_argument("--topics-output", type=Path, default=None)
    parser.add_argument("--id2topics-output", type=Path, default=None)
    parser.add_argument("--topic-info-output", type=Path, default=None)
    parser.add_argument("--log-odds-output", type=Path, default=None)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reuse-embeddings", action="store_true")
    parser.add_argument("--reuse-model", action="store_true")
    parser.add_argument("--min-topic-text-len", type=int, default=3)
    parser.add_argument("--umap-epochs", type=int, default=50_000)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--min-cluster-size", type=int, default=300)
    parser.add_argument("--max-cluster-size", type=int, default=5_000)
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--limit-docs", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--no-legacy-embeddings", action="store_true", help="Do not also write Data/embeddings.pkl.")
    parser.add_argument("--write-html", action="store_true", help="Write an interactive BERTopic document plot.")
    parser.add_argument("--html-output", type=Path, default=None)
    parser.add_argument("--visual-umap-epochs", type=int, default=1_000)
    return parser.parse_args()


def clean_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.split("/")[-1])


def prepare_docs(videos: pd.DataFrame, *, min_len: int, limit_docs: int) -> pd.DataFrame:
    videos = coerce_identifier_columns(videos)
    if "id" not in videos.columns:
        raise ValueError("Video input must contain an 'id' column.")
    text_col = "processed_text" if "processed_text" in videos.columns else "text"
    if text_col not in videos.columns:
        raise ValueError("Video input must contain 'processed_text' or 'text'.")
    docs = videos[["id", text_col] + (["create_time"] if "create_time" in videos.columns else [])].copy()
    docs = docs.rename(columns={text_col: "source_text"})
    docs["topic_text"] = [topic_preprocess_text(text) for text in tqdm(docs["source_text"], desc="Topic preprocessing")]
    docs = docs[docs["topic_text"].str.strip().str.len() >= min_len].reset_index(drop=True)
    if limit_docs > 0:
        docs = docs.head(limit_docs).copy()
    return docs


def load_or_compute_embeddings(
    docs: pd.DataFrame,
    path: Path,
    *,
    model_name: str,
    batch_size: int,
    device: str,
    reuse: bool,
) -> np.ndarray:
    if reuse and path.exists():
        with path.open("rb") as handle:
            embeddings = pickle.load(handle)
        if len(embeddings) == len(docs):
            return embeddings
        print(f"Embedding cache row mismatch ({len(embeddings)} != {len(docs)}); recomputing.")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        docs["topic_text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return embeddings


def fit_or_load_topics(
    docs: pd.DataFrame,
    embeddings: np.ndarray,
    args: argparse.Namespace,
    model_output: Path,
) -> tuple[Any, np.ndarray, Any]:
    if args.reuse_model and model_output.exists():
        with model_output.open("rb") as handle:
            topic_model = pickle.load(handle)
        topics, probs = topic_model.transform(docs["topic_text"].tolist(), embeddings=embeddings)
        return topic_model, np.asarray(topics), probs

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    import umap

    umap_model = umap.UMAP(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        min_dist=args.min_dist,
        n_epochs=args.umap_epochs,
        metric="cosine",
        random_state=args.random_state,
        verbose=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
        min_samples=args.min_samples,
        core_dist_n_jobs=1,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        verbose=True,
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
    )
    topics, probs = topic_model.fit_transform(docs["topic_text"].tolist(), embeddings=embeddings)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    with model_output.open("wb") as handle:
        pickle.dump(topic_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return topic_model, np.asarray(topics), probs


def attach_probabilities(assignments: pd.DataFrame, probs: Any) -> pd.DataFrame:
    out = assignments.copy()
    if probs is None:
        return out
    try:
        arr = np.asarray(probs)
        if arr.ndim == 1 and len(arr) == len(out):
            out["topic_probability"] = arr
        elif arr.ndim == 2 and arr.shape[0] == len(out):
            out["topic_probability"] = arr.max(axis=1)
    except Exception:
        pass
    return out


def write_interactive_html(
    topic_model: Any,
    docs: pd.DataFrame,
    embeddings: np.ndarray,
    args: argparse.Namespace,
    html_output: Path,
) -> None:
    import umap

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        n_components=2,
        min_dist=args.min_dist,
        n_epochs=args.visual_umap_epochs,
        metric="cosine",
        random_state=args.random_state,
        verbose=True,
    )
    reduced = reducer.fit_transform(embeddings)
    figure = topic_model.visualize_documents(docs["topic_text"].tolist(), reduced_embeddings=reduced)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(html_output))


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or repo_root / "Data").resolve()
    input_path = args.input or data_dir / f"videos_enriched_{args.dataset}.feather"
    docs_output = args.docs_output or data_dir / f"topic_docs_{args.dataset}.feather"
    embedding_output = args.embedding_output or data_dir / f"videos_embeddings_{args.dataset}.pkl"
    model_output = args.model_output or data_dir / f"videos_topic_model_{args.dataset}.pkl"
    assignments_output = args.assignments_output or data_dir / f"topic_assignments_{args.dataset}.feather"
    topics_output = args.topics_output or data_dir / "topics.feather"
    id2topics_output = args.id2topics_output or data_dir / "id2topics.feather"
    topic_info_output = args.topic_info_output or data_dir / f"bertopic_topic_info_{args.dataset}.csv"
    log_odds_output = args.log_odds_output or data_dir / f"topic_keywords_log_odds_{args.dataset}.csv"
    html_output = args.html_output or repo_root / "Figures" / f"bertopic_documents_{args.dataset}.html"

    videos = pd.read_feather(input_path)
    docs = prepare_docs(videos, min_len=args.min_topic_text_len, limit_docs=args.limit_docs)
    if docs.empty:
        raise ValueError("No topic documents remain after preprocessing.")
    write_feather(docs, docs_output)

    device = model_device(args.device)
    embeddings = load_or_compute_embeddings(
        docs,
        embedding_output,
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        device=device,
        reuse=args.reuse_embeddings,
    )
    if not args.no_legacy_embeddings:
        legacy_path = data_dir / "embeddings.pkl"
        with legacy_path.open("wb") as handle:
            pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    topic_model, topics, probs = fit_or_load_topics(docs, embeddings, args, model_output)
    assignments = docs.copy()
    assignments["topic_id"] = topics
    assignments = attach_probabilities(assignments, probs)
    write_feather(assignments, assignments_output)

    paper_topics = assignments[["id", "topic_id"]].rename(columns={"topic_id": "topic"})
    write_feather(paper_topics, topics_output)

    topic_info = topic_model.get_topic_info()
    write_csv(topic_info, topic_info_output)

    log_terms, id2topics = log_odds_keywords(assignments, topic_col="topic_id", text_col="topic_text", top_n=10)
    write_csv(log_terms, log_odds_output)
    write_feather(id2topics, id2topics_output)

    if args.write_html:
        write_interactive_html(topic_model, docs, embeddings, args, html_output)

    print(f"Topic docs: {docs_output} ({len(docs):,} rows)")
    print(f"Embeddings: {embedding_output}")
    if not args.no_legacy_embeddings:
        print(f"Legacy embeddings: {data_dir / 'embeddings.pkl'}")
    print(f"Model: {model_output}")
    print(f"Assignments: {assignments_output}")
    print(f"Paper topic files: {topics_output}, {id2topics_output}")
    print(f"Topic info: {topic_info_output}")
    print(f"Log-odds keywords: {log_odds_output}")
    if args.write_html:
        print(f"Interactive HTML: {html_output}")
    print(f"Embedding model slug: {clean_slug(args.embedding_model)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
