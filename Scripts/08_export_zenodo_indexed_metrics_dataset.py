from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiktokresearch import repo_root_from_script


VARIANT_SLUG = "paper_locked_multilingual_detoxify_mpnet"
MIN_ENTRIES_PER_DATE = 10

TOXICITY_COLUMNS = [
    "detoxify_toxicity",
    "detoxify_severe_toxicity",
    "detoxify_obscene",
    "detoxify_identity_attack",
    "detoxify_insult",
    "detoxify_threat",
    "detoxify_sexual_explicit",
]

SENTIMENT_COLUMNS = [
    "xlm_sentiment",
    "xlm_polarity",
    "xlm_negative",
    "xlm_neutral",
    "xlm_positive",
]

VIDEO_COLUMNS = ["topic", "video_index", "date", *TOXICITY_COLUMNS, *SENTIMENT_COLUMNS]
COMMENT_COLUMNS = ["video_index", "comment_index", "date", *TOXICITY_COLUMNS, *SENTIMENT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the Zenodo-shareable indexed metrics dataset for the paper-locked "
            "multilingual Detoxify/all-mpnet-base-v2 variant."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument(
        "--variant-dir",
        type=Path,
        default=None,
        help=(
            "Variant directory containing Data/videos_analysis.feather and "
            "Data/comments_raw_topic_valid.feather. Defaults to "
            "<repo-root>/old/Variants/paper_locked_multilingual_detoxify_mpnet."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <repo-root>/zenodo/indexed_metrics_dataset.",
    )
    parser.add_argument(
        "--min-entries-per-date",
        type=int,
        default=MIN_ENTRIES_PER_DATE,
        help="Suppress dates with fewer than this many total video/comment rows.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def utc_date_column(create_time: pd.Series) -> pd.Series:
    timestamp = pd.to_numeric(create_time, errors="coerce").astype("Int64")
    datetime_utc = pd.to_datetime(timestamp, unit="s", utc=True, errors="coerce")
    return datetime_utc.dt.strftime("%Y-%m-%d")


def filter_sparse_dates(
    videos: pd.DataFrame,
    comments: pd.DataFrame,
    min_entries_per_date: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    videos = videos.copy()
    comments = comments.copy()

    while True:
        date_counts = pd.concat([videos["date"], comments["date"]], ignore_index=True).value_counts()
        valid_dates = set(date_counts[date_counts >= min_entries_per_date].index)

        filtered_videos = videos[videos["date"].isin(valid_dates)].copy()
        valid_video_ids = set(filtered_videos["id"])
        filtered_comments = comments[
            comments["date"].isin(valid_dates) & comments["video_id"].isin(valid_video_ids)
        ].copy()

        if len(filtered_videos) == len(videos) and len(filtered_comments) == len(comments):
            return filtered_videos.reset_index(drop=True), filtered_comments.reset_index(drop=True)

        videos = filtered_videos
        comments = filtered_comments


def build_exports(
    videos: pd.DataFrame,
    comments: pd.DataFrame,
    min_entries_per_date: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        videos,
        ["id", "topic_name", "create_time", *TOXICITY_COLUMNS, *SENTIMENT_COLUMNS],
        Path("videos_analysis.feather"),
    )
    require_columns(
        comments,
        ["id", "video_id", "create_time", *TOXICITY_COLUMNS, *SENTIMENT_COLUMNS],
        Path("comments_raw_topic_valid.feather"),
    )

    videos = videos.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True).copy()
    comments = comments.drop_duplicates(subset=["id", "video_id"], keep="first").reset_index(drop=True).copy()

    videos["id"] = videos["id"].astype(str)
    comments["video_id"] = comments["video_id"].astype(str)
    videos["date"] = utc_date_column(videos["create_time"])
    comments["date"] = utc_date_column(comments["create_time"])

    videos, comments = filter_sparse_dates(videos, comments, min_entries_per_date)

    video_id_to_index = pd.Series(range(len(videos)), index=videos["id"]).to_dict()
    missing_parent_mask = ~comments["video_id"].isin(video_id_to_index)
    if missing_parent_mask.any():
        missing = int(missing_parent_mask.sum())
        raise ValueError(f"{missing} comments reference videos outside videos_analysis.feather")

    video_export = videos[["topic_name", "date", *TOXICITY_COLUMNS, *SENTIMENT_COLUMNS]].copy()
    video_export.insert(0, "video_index", range(len(video_export)))
    video_export = video_export.rename(columns={"topic_name": "topic"})
    video_export = video_export[VIDEO_COLUMNS]

    comment_export = comments[["date", *TOXICITY_COLUMNS, *SENTIMENT_COLUMNS]].copy()
    comment_export.insert(0, "comment_index", range(len(comment_export)))
    comment_export.insert(0, "video_index", comments["video_id"].map(video_id_to_index).astype("int64"))
    comment_export = comment_export[COMMENT_COLUMNS]

    return video_export, comment_export


def validate_exports(
    video_export: pd.DataFrame,
    comment_export: pd.DataFrame,
    min_entries_per_date: int,
) -> None:
    disallowed = {
        "id",
        "video_id",
        "comment_id",
        "parent_comment_id",
        "username",
        "author",
        "text",
        "processed_text",
        "video_description",
        "voice_to_text",
    }
    leaked_video_cols = disallowed.intersection(video_export.columns)
    leaked_comment_cols = disallowed.intersection(comment_export.columns)
    if leaked_video_cols or leaked_comment_cols:
        raise ValueError(
            "Export contains disallowed columns: "
            f"videos={sorted(leaked_video_cols)}, comments={sorted(leaked_comment_cols)}"
        )

    if list(video_export.columns) != VIDEO_COLUMNS:
        raise ValueError(f"Unexpected videos.csv columns: {list(video_export.columns)}")
    if list(comment_export.columns) != COMMENT_COLUMNS:
        raise ValueError(f"Unexpected comments.csv columns: {list(comment_export.columns)}")
    if not video_export["video_index"].is_unique:
        raise ValueError("video_index is not unique in videos.csv")
    if not comment_export["comment_index"].is_unique:
        raise ValueError("comment_index is not unique in comments.csv")
    if not comment_export["video_index"].isin(video_export["video_index"]).all():
        raise ValueError("Some comment video_index values are absent from videos.csv")

    date_counts = pd.concat([video_export["date"], comment_export["date"]], ignore_index=True).value_counts()
    sparse_dates = date_counts[date_counts < min_entries_per_date]
    if not sparse_dates.empty:
        raise ValueError(f"Export contains dates below threshold: {sparse_dates.to_dict()}")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    variant_dir = (args.variant_dir or repo_root / "old" / "Variants" / VARIANT_SLUG).resolve()
    output_dir = (args.output_dir or repo_root / "zenodo" / "indexed_metrics_dataset").resolve()

    data_dir = variant_dir / "Data"
    videos_path = data_dir / "videos_analysis.feather"
    comments_path = data_dir / "comments_raw_topic_valid.feather"
    if not videos_path.exists():
        raise FileNotFoundError(videos_path)
    if not comments_path.exists():
        raise FileNotFoundError(comments_path)

    videos = pd.read_feather(videos_path)
    comments = pd.read_feather(comments_path)
    video_export, comment_export = build_exports(videos, comments, args.min_entries_per_date)
    validate_exports(video_export, comment_export, args.min_entries_per_date)

    output_dir.mkdir(parents=True, exist_ok=True)
    videos_output = output_dir / "videos.csv"
    comments_output = output_dir / "comments.csv"
    video_export.to_csv(videos_output, index=False)
    comment_export.to_csv(comments_output, index=False)

    print(f"Wrote {len(video_export):,} videos -> {videos_output}")
    print(f"Wrote {len(comment_export):,} comments -> {comments_output}")
    print(f"Minimum combined rows per exported date: {args.min_entries_per_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
