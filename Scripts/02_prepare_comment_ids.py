from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiktokresearch import (
    DATASET_DEFAULT,
    coerce_identifier_columns,
    read_video_jsons,
    repo_root_from_script,
    write_csv,
    write_feather,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine collected video JSON files and create a video-id list for comment collection."
    )
    parser.add_argument("--dataset", default=DATASET_DEFAULT)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--data-dir", type=Path, default=None, help="Defaults to <repo-root>/Data")
    parser.add_argument("--collected-dir", type=Path, default=None, help="Defaults to <data-dir>/collected")
    parser.add_argument("--pattern", default=None, help="Glob pattern for collected JSON files.")
    parser.add_argument("--video-input", type=Path, default=None, help="Optional feather/csv/json video file.")
    parser.add_argument("--videos-output", type=Path, default=None)
    parser.add_argument("--ids-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--min-comment-count", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output IDs.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_video_input(path: Path) -> pd.DataFrame:
    if path.suffix == ".feather":
        return pd.read_feather(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("results", payload if isinstance(payload, list) else [])
        return pd.DataFrame(records)
    raise ValueError(f"Unsupported video input: {path}")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or repo_root / "Data").resolve()
    collected_dir = (args.collected_dir or data_dir / "collected").resolve()
    videos_output = args.videos_output or data_dir / f"videos_raw_{args.dataset}.feather"
    ids_output = args.ids_output or data_dir / f"{args.dataset}_video_ids_for_comments.csv"
    summary_output = args.summary_output or data_dir / f"collection_summary_{args.dataset}.csv"
    pattern = args.pattern or f"{args.dataset}*.json"

    if args.video_input:
        videos = read_video_input(args.video_input)
    else:
        videos = read_video_jsons(collected_dir, pattern)
    if videos.empty:
        raise FileNotFoundError(f"No videos found from {args.video_input or collected_dir / pattern}")

    videos = coerce_identifier_columns(videos)
    if "id" not in videos.columns:
        raise ValueError("Video data must contain an 'id' column.")
    if "comment_count" not in videos.columns:
        raise ValueError("Video data must contain a 'comment_count' column.")

    videos["comment_count"] = pd.to_numeric(videos["comment_count"], errors="coerce").fillna(0).astype(int)
    videos_unique = videos.drop_duplicates(subset=["id"], keep="first").copy()
    eligible = videos_unique[videos_unique["comment_count"] >= args.min_comment_count].copy()
    ids = eligible[["id"]].rename(columns={"id": "video_id"}).dropna().drop_duplicates()
    if args.shuffle:
        ids = ids.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

    summary_rows = [
        {"measure": "video_rows", "value": len(videos)},
        {"measure": "unique_video_ids", "value": videos["id"].nunique()},
        {"measure": "videos_after_dedup", "value": len(videos_unique)},
        {"measure": f"videos_comment_count_ge_{args.min_comment_count}", "value": len(eligible)},
        {"measure": "video_ids_for_comments", "value": len(ids)},
    ]
    if "source_file" in videos.columns:
        summary_rows.append({"measure": "source_files", "value": videos["source_file"].nunique()})

    write_feather(videos_unique, videos_output)
    write_csv(ids, ids_output)
    write_csv(pd.DataFrame(summary_rows), summary_output)

    print(f"Videos raw: {videos_output} ({len(videos_unique):,} unique videos)")
    print(f"Comment ID list: {ids_output} ({len(ids):,} video IDs)")
    print(f"Summary: {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
