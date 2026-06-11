from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiktokresearch import DATASET_DEFAULT, ensure_xlm_polarity, model_device, repo_root_from_script, write_feather


XLM_COLUMNS = ["xlm_negative", "xlm_neutral", "xlm_positive", "xlm_sentiment", "xlm_polarity"]
VADER_COLUMNS = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XLM-T sentiment, VADER, and Detoxify on TikTok text.")
    parser.add_argument("--dataset", default=DATASET_DEFAULT)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--data-dir", type=Path, default=None, help="Defaults to <repo-root>/Data")
    parser.add_argument("--videos-input", type=Path, default=None)
    parser.add_argument("--comments-input", type=Path, default=None)
    parser.add_argument("--videos-output", type=Path, default=None)
    parser.add_argument("--comments-output", type=Path, default=None)
    parser.add_argument("--text-col", default="processed_text")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detoxify-model", default="unbiased", choices=["original", "unbiased", "multilingual"])
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-comments", action="store_true")
    parser.add_argument("--skip-xlm", action="store_true")
    parser.add_argument("--skip-vader", action="store_true")
    parser.add_argument("--skip-detoxify", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--no-redacted", action="store_true", help="Do not write legacy redacted companion files.")
    return parser.parse_args()


def default_input(data_dir: Path, dataset: str, kind: str) -> Path:
    preprocessed = data_dir / f"{kind}_preprocessed_{dataset}.feather"
    if preprocessed.exists():
        return preprocessed
    return data_dir / f"{kind}_enriched_{dataset}.feather"


def has_reusable_scores(path: Path, *, need_xlm: bool, need_vader: bool, need_detoxify: bool) -> bool:
    if not path.exists():
        return False
    columns = set(pd.read_feather(path).columns)
    if need_xlm and not set(XLM_COLUMNS).issubset(columns):
        return False
    if need_vader and not set(VADER_COLUMNS).issubset(columns):
        return False
    if need_detoxify and not any(col.startswith("detoxify_") for col in columns):
        return False
    return True


def run_xlm_sentiment(df: pd.DataFrame, *, text_col: str, batch_size: int, device: str) -> pd.DataFrame:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    texts = df[text_col].fillna("").astype(str).tolist()
    batches: list[np.ndarray] = []
    max_len = min(getattr(model.config, "max_position_embeddings", 514), 512)

    for start in tqdm(range(0, len(texts), batch_size), desc="XLM-T sentiment"):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**tokens).logits, dim=1).detach().cpu().numpy()
        batches.append(probs)

    arr = np.vstack(batches) if batches else np.empty((0, 3))
    out = df.drop(columns=[col for col in XLM_COLUMNS if col in df.columns], errors="ignore").copy()
    out["xlm_negative"] = arr[:, 0]
    out["xlm_neutral"] = arr[:, 1]
    out["xlm_positive"] = arr[:, 2]
    labels = np.array(["xlm_negative", "xlm_neutral", "xlm_positive"])
    out["xlm_sentiment"] = labels[np.argmax(arr, axis=1)] if len(arr) else []
    out["xlm_polarity"] = out["xlm_positive"] - out["xlm_negative"]
    return out


def run_vader(df: pd.DataFrame, *, text_col: str) -> pd.DataFrame:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    nltk.download("vader_lexicon", quiet=True)
    analyzer = SentimentIntensityAnalyzer()
    scores = df[text_col].fillna("").astype(str).map(analyzer.polarity_scores).apply(pd.Series)
    scores = scores.rename(columns={"neg": "vader_neg", "neu": "vader_neu", "pos": "vader_pos", "compound": "vader_compound"})
    out = df.drop(columns=[col for col in VADER_COLUMNS if col in df.columns], errors="ignore").copy()
    return pd.concat([out, scores[VADER_COLUMNS]], axis=1)


def run_detoxify(df: pd.DataFrame, *, text_col: str, batch_size: int, model_name: str, device: str) -> pd.DataFrame:
    from detoxify import Detoxify

    try:
        model = Detoxify(model_name, device=device)
    except TypeError:
        model = Detoxify(model_name)

    texts = df[text_col].fillna("").astype(str).tolist()
    results: dict[str, list[float]] = {}
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Detoxify({model_name})"):
        batch = texts[start : start + batch_size]
        prediction = model.predict(batch)
        for key, values in prediction.items():
            results.setdefault(f"detoxify_{key}", []).extend(float(value) for value in values)

    out = df.drop(columns=[col for col in df.columns if col.startswith("detoxify_")], errors="ignore").copy()
    for key, values in results.items():
        out[key] = values
    return out


def score_frame(df: pd.DataFrame, args: argparse.Namespace, *, label: str) -> pd.DataFrame:
    if args.text_col not in df.columns:
        raise ValueError(f"{label} input is missing text column {args.text_col!r}")
    if args.limit_rows > 0:
        df = df.head(args.limit_rows).copy()
    device = model_device(args.device)
    out = df.copy()
    if not args.skip_xlm:
        out = run_xlm_sentiment(out, text_col=args.text_col, batch_size=args.batch_size, device=device)
    else:
        out = ensure_xlm_polarity(out)
    if not args.skip_vader:
        out = run_vader(out, text_col=args.text_col)
    if not args.skip_detoxify:
        out = run_detoxify(
            out,
            text_col=args.text_col,
            batch_size=args.batch_size,
            model_name=args.detoxify_model,
            device=device,
        )
    return out


def write_redacted(data_dir: Path, dataset: str, kind: str, df: pd.DataFrame) -> None:
    if kind == "videos":
        drop_cols = ["id", "text", "processed_text", "username", "video_description", "voice_to_text", "source_file"]
        nocomment_path = data_dir / f"videos_enriched_nocomment_{dataset}.feather"
        noref_path = data_dir / f"videos_enriched_{dataset}_noref.feather"
    else:
        drop_cols = ["id", "text", "processed_text", "username", "video_description", "voice_to_text", "source_file"]
        nocomment_path = data_dir / f"comments_enriched_nocomment_{dataset}.feather"
        noref_path = data_dir / f"comments_enriched_{dataset}_noref.feather"
    write_feather(df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore"), nocomment_path)
    write_feather(df, noref_path)


def process_kind(args: argparse.Namespace, data_dir: Path, *, kind: str, input_path: Path, output_path: Path) -> None:
    need_xlm = not args.skip_xlm
    need_vader = not args.skip_vader
    need_detoxify = not args.skip_detoxify
    if args.reuse and has_reusable_scores(output_path, need_xlm=need_xlm, need_vader=need_vader, need_detoxify=need_detoxify):
        print(f"Reusing {output_path}")
        return
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    df = pd.read_feather(input_path)
    scored = score_frame(df, args, label=kind)
    write_feather(scored, output_path)
    if not args.no_redacted:
        write_redacted(data_dir, args.dataset, kind, scored)
    print(f"{kind.capitalize()} enriched: {output_path} ({len(scored):,} rows)")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or repo_root / "Data").resolve()
    videos_input = args.videos_input or default_input(data_dir, args.dataset, "videos")
    comments_input = args.comments_input or default_input(data_dir, args.dataset, "comments")
    videos_output = args.videos_output or data_dir / f"videos_enriched_{args.dataset}.feather"
    comments_output = args.comments_output or data_dir / f"comments_enriched_{args.dataset}.feather"

    if not args.skip_videos:
        process_kind(args, data_dir, kind="videos", input_path=videos_input, output_path=videos_output)
    if not args.skip_comments:
        process_kind(args, data_dir, kind="comments", input_path=comments_input, output_path=comments_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
