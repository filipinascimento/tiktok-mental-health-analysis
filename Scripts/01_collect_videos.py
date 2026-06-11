from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiktokresearch import (
    DATASET_DEFAULT,
    VIDEO_FIELDS,
    TikTokAPI,
    api_date,
    build_research_query,
    date_window_for_year,
    iter_dates_inclusive,
    load_terms,
    parse_csv_list,
    repo_root_from_script,
    response_payload,
    write_json_atomically,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect TikTok Research API videos for the Mental Health Month pipeline."
    )
    parser.add_argument("--dataset", default=DATASET_DEFAULT)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--data-dir", type=Path, default=None, help="Defaults to <repo-root>/Data")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <data-dir>/collected")
    parser.add_argument("--error-dir", type=Path, default=None, help="Defaults to <data-dir>/errors")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--window-start", default="04-15", help="MM-DD start date for each year")
    parser.add_argument("--window-end", default="06-15", help="MM-DD inclusive end date for each year")
    parser.add_argument("--terms", default=None, help="Comma-separated query terms. Defaults to paper term list.")
    parser.add_argument("--terms-file", type=Path, default=None, help="One query term per line.")
    parser.add_argument("--regions", default="", help="Comma-separated TikTok region codes, e.g. US,CA.")
    parser.add_argument("--runs", type=int, default=1, help="Independent runs per day.")
    parser.add_argument("--max-count", type=int, default=100_000)
    parser.add_argument("--count-per-page", type=int, default=100)
    parser.add_argument("--ratelimit", type=float, default=0.2, help="Requests per second.")
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--prefix", default=None, help="Filename prefix. Defaults to --dataset.")
    parser.add_argument("--no-random", action="store_true", help="Use sequential API results instead of random sampling.")
    parser.add_argument("--no-keyword", action="store_true", help="Do not query the keyword field.")
    parser.add_argument("--no-hashtag", action="store_true", help="Do not query the hashtag_name field.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip day/run files already present.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned queries without calling the API.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def existing_run(output_dir: Path, error_dir: Path, prefix: str, query_date: str, run_index: int) -> bool:
    glob_name = f"{prefix}_*_{query_date}_run{run_index:02d}_*.json"
    return any(output_dir.glob(glob_name)) or any(error_dir.glob(glob_name))


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or repo_root / "Data").resolve()
    output_dir = (args.output_dir or data_dir / "collected").resolve()
    error_dir = (args.error_dir or data_dir / "errors").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    terms = load_terms(args.terms, args.terms_file)
    regions = parse_csv_list(args.regions)
    include_keyword = not args.no_keyword
    include_hashtag = not args.no_hashtag
    query = build_research_query(terms, include_keyword=include_keyword, include_hashtag=include_hashtag, regions=regions)
    prefix = args.prefix or args.dataset
    random_flag = not args.no_random
    mode = "random" if random_flag else "sequential"
    scope = "_".join(regions) if regions else "global"

    planned: list[tuple[int, str, int]] = []
    for year in args.years:
        start, end = date_window_for_year(year, args.window_start, args.window_end)
        for current in iter_dates_inclusive(start, end):
            date_string = api_date(current)
            for run_index in range(1, args.runs + 1):
                if not args.no_resume and existing_run(output_dir, error_dir, prefix, date_string, run_index):
                    continue
                planned.append((year, date_string, run_index))

    print(f"Dataset: {args.dataset}")
    print(f"Terms ({len(terms)}): {', '.join(terms)}")
    print(f"Regions: {', '.join(regions) if regions else 'global'}")
    print(f"Planned API calls: {len(planned)}")
    if args.dry_run:
        for year, date_string, run_index in planned[:20]:
            print(f"DRY {year} {date_string} run={run_index}")
        if len(planned) > 20:
            print(f"... {len(planned) - 20} more")
        return 0

    api = TikTokAPI(
        os.environ.get("TIKTOK_CLIENT_KEY", ""),
        os.environ.get("TIKTOK_CLIENT_SECRET", ""),
        ratelimit=args.ratelimit,
    )

    for year, date_string, run_index in tqdm(planned, desc="Video query runs"):
        query_metadata = {
            "dataset": args.dataset,
            "query": query,
            "terms": terms,
            "regions": regions,
            "year": year,
            "start_date": date_string,
            "end_date": date_string,
            "run_index": run_index,
            "runs": args.runs,
            "is_random": random_flag,
            "max_count": args.max_count,
            "count_per_page": args.count_per_page,
            "fields": list(VIDEO_FIELDS),
        }

        results = []
        response = None
        error = None
        for attempt in range(2):
            results, response, error = api.query_videos(
                query,
                fields=VIDEO_FIELDS,
                start_date=date_string,
                end_date=date_string,
                is_random=random_flag,
                max_count=args.max_count,
                count_per_page=args.count_per_page,
                max_trials=args.max_trials,
                verbose=args.verbose,
                show_progress=False,
            )
            if response is not None and response.status_code == 401 and attempt == 0:
                api.refresh_token()
                continue
            break

        target_dir = output_dir if error is None else error_dir
        file_name = f"{prefix}_{mode}_{scope}_{date_string}_run{run_index:02d}_{uuid.uuid1()}.json"
        payload = {
            "results": results,
            "error": str(error) if error else None,
            "response": response_payload(response),
            "query": query_metadata,
        }
        write_json_atomically(target_dir / file_name, payload)
        status = getattr(response, "status_code", None)
        print(f"{date_string} run={run_index}: {len(results):,} videos status={status} -> {target_dir / file_name}")

    print(f"Collected files: {output_dir}")
    print(f"Error files: {error_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
