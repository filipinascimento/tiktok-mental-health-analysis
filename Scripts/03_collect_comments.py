from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiktokresearch import (
    COMMENT_FIELDS,
    DATASET_DEFAULT,
    TikTokAPI,
    repo_root_from_script,
    write_json_atomically,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect comments for a prepared list of TikTok video IDs.")
    parser.add_argument("--dataset", default=DATASET_DEFAULT)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--data-dir", type=Path, default=None, help="Defaults to <repo-root>/Data")
    parser.add_argument("--video-ids-file", type=Path, default=None)
    parser.add_argument("--save-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--max-count", type=int, default=100_000)
    parser.add_argument("--count-per-page", type=int, default=100)
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--ratelimit", type=float, default=0.2)
    parser.add_argument("--rate-limit-sleep", type=int, default=3600)
    parser.add_argument("--limit-videos", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_video_ids(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "video_id" in df.columns:
        values = df["video_id"]
    else:
        values = df.iloc[:, 0]
    return [str(value) for value in values.dropna().drop_duplicates().tolist()]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def next_batch_index(batch_dir: Path) -> int:
    existing = sorted(batch_dir.glob("comments_batch_*.jsonl"))
    if not existing:
        return 0
    return max(int(path.stem.split("_")[-1]) for path in existing) + 1


class JsonlBatchWriter:
    def __init__(self, batch_dir: Path, batch_index: int, batch_count: int, batch_size: int) -> None:
        self.batch_dir = batch_dir
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.batch_index = batch_index
        self.batch_count = batch_count
        self.batch_size = batch_size
        self.handle = self._open_current_file()

    def _path(self) -> Path:
        return self.batch_dir / f"comments_batch_{self.batch_index:05d}.jsonl"

    def _open_current_file(self):
        return self._path().open("a", encoding="utf-8")

    def write_comments(self, video_id: str, comments: list[dict[str, Any]]) -> int:
        written = 0
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            if not comment.get("video_id"):
                comment["video_id"] = video_id
            self.handle.write(json.dumps(comment, ensure_ascii=False))
            self.handle.write("\n")
            self.batch_count += 1
            written += 1
            if self.batch_count >= self.batch_size:
                self.rotate()
        return written

    def rotate(self) -> None:
        self.flush()
        self.handle.close()
        self.batch_index += 1
        self.batch_count = 0
        self.handle = self._open_current_file()

    def flush(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        if not self.handle.closed:
            self.flush()
            self.handle.close()

    def state(self) -> tuple[int, int]:
        return self.batch_index, self.batch_count


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or repo_root / "Data").resolve()
    video_ids_file = args.video_ids_file or data_dir / f"{args.dataset}_video_ids_for_comments.csv"
    save_root = args.save_root or data_dir / f"collectedComments_{args.dataset}"
    batch_dir = save_root / "comment_batches"
    errors_file = save_root / "errors.json"
    state_file = save_root / "state.json"
    save_root.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run and not video_ids_file.exists():
        print(f"Would read video IDs from missing file: {video_ids_file}")
        print(f"Output root: {save_root}")
        return 0

    video_ids = load_video_ids(video_ids_file)
    if args.limit_videos > 0:
        video_ids = video_ids[: args.limit_videos]
    if args.dry_run:
        print(f"Would collect comments for {len(video_ids):,} videos from {video_ids_file}")
        print(f"Output root: {save_root}")
        return 0

    if args.no_resume:
        processed_videos: set[str] = set()
        batch_index = next_batch_index(batch_dir)
        batch_count = 0
    else:
        state = load_json(state_file, {})
        processed_videos = set(state.get("processed_videos", []))
        batch_index = int(state.get("batch_index", 0))
        batch_count = int(state.get("batch_count", 0))
    errors: dict[str, Any] = load_json(errors_file, {})

    remaining = [video_id for video_id in video_ids if video_id not in processed_videos]
    writer = JsonlBatchWriter(batch_dir, batch_index, batch_count, args.batch_size)
    api = TikTokAPI(
        os.environ.get("TIKTOK_CLIENT_KEY", ""),
        os.environ.get("TIKTOK_CLIENT_SECRET", ""),
        ratelimit=args.ratelimit,
    )

    def save_state() -> None:
        current_batch, current_count = writer.state()
        write_json_atomically(
            state_file,
            {
                "processed_videos": sorted(processed_videos),
                "batch_index": current_batch,
                "batch_count": current_count,
                "updated_at": datetime.utcnow().isoformat(),
            },
        )

    save_state()
    print(f"Loaded {len(video_ids):,} video IDs; {len(processed_videos):,} already processed; {len(remaining):,} remaining")

    progress = tqdm(total=len(video_ids), initial=len(processed_videos), desc="Videos")
    try:
        for video_id in remaining:
            response = None
            error = None
            comments: list[dict[str, Any]] = []
            try:
                while True:
                    comments, response, error = api.query_comments(
                        video_id,
                        fields=COMMENT_FIELDS,
                        count_per_page=args.count_per_page,
                        max_count=args.max_count,
                        max_trials=args.max_trials,
                        show_progress=False,
                    )
                    status = getattr(response, "status_code", None)
                    if status == 401:
                        api.refresh_token()
                        continue
                    if status == 429:
                        print(f"Rate limit while fetching {video_id}; sleeping {args.rate_limit_sleep}s")
                        time.sleep(args.rate_limit_sleep)
                        continue
                    break

                status = getattr(response, "status_code", None)
                if error is not None or status != 200:
                    errors[video_id] = {
                        "status_code": status,
                        "message": str(error) if error else "non-200 response",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    write_json_atomically(errors_file, errors)
                    save_state()
                else:
                    written = writer.write_comments(video_id, comments or [])
                    writer.flush()
                    processed_videos.add(video_id)
                    errors.pop(video_id, None)
                    write_json_atomically(errors_file, errors)
                    save_state()
                    print(f"{video_id}: {written:,} comments")
                progress.update(1)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                errors[video_id] = {
                    "status_code": getattr(response, "status_code", None),
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                write_json_atomically(errors_file, errors)
                save_state()
                progress.update(1)
    finally:
        progress.close()
        save_state()
        writer.close()

    print(f"Comment batches: {batch_dir}")
    print(f"Processed videos: {len(processed_videos):,}")
    print(f"Errors: {len(errors):,} ({errors_file})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
