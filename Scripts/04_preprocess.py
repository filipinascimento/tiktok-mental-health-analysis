from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiktokresearch import (
    DATASET_DEFAULT,
    coerce_identifier_columns,
    compose_video_text,
    normalize_social_text,
    read_comment_batches,
    read_video_jsons,
    repo_root_from_script,
    write_csv,
    write_feather,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess collected TikTok videos and comments.")
    parser.add_argument("--dataset", default=DATASET_DEFAULT)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--data-dir", type=Path, default=None, help="Defaults to <repo-root>/Data")
    parser.add_argument("--videos-input", type=Path, default=None)
    parser.add_argument("--comments-input", type=Path, default=None)
    parser.add_argument("--collected-dir", type=Path, default=None)
    parser.add_argument("--comment-batch-dir", type=Path, default=None)
    parser.add_argument("--video-pattern", default=None)
    parser.add_argument("--videos-output", type=Path, default=None)
    parser.add_argument("--comments-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--no-hashtags", action="store_true", help="Do not append hashtags to video text.")
    parser.add_argument("--min-comment-chars", type=int, default=0)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".feather":
        return pd.read_feather(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported table input: {path}")


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_videos(args: argparse.Namespace, data_dir: Path) -> pd.DataFrame:
    if args.videos_input:
        return read_table(args.videos_input)

    existing = first_existing(
        [
            data_dir / f"videos_raw_{args.dataset}.feather",
            data_dir / f"videos_enriched_{args.dataset}.feather",
        ]
    )
    if existing:
        return read_table(existing)

    collected_dir = args.collected_dir or data_dir / "collected"
    pattern = args.video_pattern or f"{args.dataset}*.json"
    return read_video_jsons(collected_dir, pattern)


def load_comments(args: argparse.Namespace, data_dir: Path) -> pd.DataFrame:
    if args.comments_input:
        return read_table(args.comments_input)

    existing = first_existing(
        [
            data_dir / f"comments_raw_{args.dataset}.feather",
            data_dir / f"comments_enriched_{args.dataset}.feather",
        ]
    )
    if existing:
        return read_table(existing)

    batch_dir = args.comment_batch_dir or data_dir / f"collectedComments_{args.dataset}" / "comment_batches"
    return read_comment_batches(batch_dir)


def preprocess_videos(videos: pd.DataFrame, *, include_hashtags: bool, dedupe: bool) -> pd.DataFrame:
    if videos.empty:
        raise ValueError("No video rows found.")
    videos = coerce_identifier_columns(videos)
    if "id" not in videos.columns:
        raise ValueError("Videos must contain an 'id' column.")
    videos["text"] = videos.apply(lambda row: compose_video_text(row, include_hashtags=include_hashtags), axis=1)
    videos["processed_text"] = videos["text"].map(normalize_social_text)
    if dedupe:
        videos = videos.drop_duplicates(subset=["id"], keep="first").copy()
    return videos.reset_index(drop=True)


def preprocess_comments(comments: pd.DataFrame, *, dedupe: bool, min_chars: int) -> pd.DataFrame:
    if comments.empty:
        raise ValueError("No comment rows found.")
    comments = coerce_identifier_columns(comments)
    if "id" not in comments.columns or "video_id" not in comments.columns:
        raise ValueError("Comments must contain 'id' and 'video_id' columns.")
    if "text" not in comments.columns:
        raise ValueError("Comments must contain a 'text' column.")
    comments["processed_text"] = comments["text"].map(normalize_social_text)
    if min_chars > 0:
        signal = comments["processed_text"].fillna("").astype(str).str.count(r"[A-Za-z0-9]")
        comments = comments[signal >= min_chars].copy()
    if dedupe:
        comments = comments.drop_duplicates(subset=["id", "video_id"], keep="first").copy()
    return comments.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or repo_root / "Data").resolve()
    videos_output = args.videos_output or data_dir / f"videos_preprocessed_{args.dataset}.feather"
    comments_output = args.comments_output or data_dir / f"comments_preprocessed_{args.dataset}.feather"
    summary_output = args.summary_output or data_dir / f"preprocess_summary_{args.dataset}.csv"

    videos_raw = load_videos(args, data_dir)
    comments_raw = load_comments(args, data_dir)
    videos = preprocess_videos(videos_raw, include_hashtags=not args.no_hashtags, dedupe=not args.no_dedupe)
    comments = preprocess_comments(comments_raw, dedupe=not args.no_dedupe, min_chars=args.min_comment_chars)

    summary = pd.DataFrame(
        [
            {"measure": "video_rows_in", "value": len(videos_raw)},
            {"measure": "video_rows_out", "value": len(videos)},
            {"measure": "unique_video_ids_out", "value": videos["id"].nunique()},
            {"measure": "comment_rows_in", "value": len(comments_raw)},
            {"measure": "comment_rows_out", "value": len(comments)},
            {"measure": "unique_comment_id_video_out", "value": len(comments.drop_duplicates(subset=["id", "video_id"]))},
        ]
    )

    write_feather(videos, videos_output)
    write_feather(comments, comments_output)
    write_csv(summary, summary_output)

    print(f"Videos preprocessed: {videos_output} ({len(videos):,} rows)")
    print(f"Comments preprocessed: {comments_output} ({len(comments):,} rows)")
    print(f"Summary: {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
