from __future__ import annotations

import ast
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import requests
from tqdm.auto import tqdm


DATASET_DEFAULT = "mentalhealth2024"

# The paper text contains "mentalilness"; the old 2024 collection script used
# "mentalillness". Keep both to make the collection script explicit and auditable.
MENTAL_HEALTH_TERMS = [
    "MentalHealthAwareness",
    "anxiety",
    "mentalhealth",
    "depression",
    "mentalhealthawarenessmonth",
    "mentalhealthmatters",
    "mentalillness",
    "mentalilness",
    "stress",
    "suicide",
    "adhd",
    "burnout",
    "trauma",
    "suicideprevention",
    "emotionalwellbeing",
]

VIDEO_FIELDS = [
    "id",
    "video_description",
    "create_time",
    "region_code",
    "share_count",
    "view_count",
    "like_count",
    "comment_count",
    "music_id",
    "hashtag_names",
    "username",
    "effect_ids",
    "playlist_id",
    "voice_to_text",
    "favorites_count",
]

COMMENT_FIELDS = [
    "id",
    "video_id",
    "text",
    "like_count",
    "reply_count",
    "parent_comment_id",
    "create_time",
]

ID_COLUMNS = {
    "id",
    "video_id",
    "music_id",
    "parent_comment_id",
    "playlist_id",
}

TOPIC_MODEL_DOMAIN_STOPWORDS = {
    "mentalhealth",
    "mentalhealthawareness",
    "mentalhealthmatters",
    "mentalhealthawarenessmonth",
    "mentalillness",
    "mentalhealthawarness",
    "mentalhealthtiktok",
    "mentalhealthtiktoks",
    "mentalhealthawarenessweek",
    "mental",
    "health",
    "awareness",
    "month",
    "may",
    "dont",
    "matters",
    "mentalhealthmonth",
    "mentalillnessawareness",
    "mentalillnessawarenessweek",
}

TOPIC_MODEL_SOCIAL_STOPWORDS = {
    "page",
    "for",
    "fyp",
    "fy",
    "fypviral",
    "foryou",
    "foryourpage",
    "tiktok",
    "capcut",
    "xyzbca",
    "sweden",
    "viralvideos",
    "viral",
    "trending",
    "outpatient",
    "foryoupage",
    "fypage",
    "fyppppppppppppppppppppppp",
    "viraltiktok",
    "greenscreenvideo",
    "greenscreen",
    "youre",
    "like",
    "im",
    "know",
    "get",
    "um",
    "xzybca",
    "tw",
    "xyzcba",
    "really",
    "people",
    "things",
    "viralvideo",
    "ive",
    "yeah",
    "painhub",
    "fyfyfy",
    "fr",
    "mh",
    "sh",
    "foryourepage",
    "duet",
    "duets",
    "dueto",
    "duett",
    "duetme",
    "duetwith",
    "duetwithme",
    "duetwithuser",
    "duetchain",
    "duetcrashers",
    "duetotiktokers",
    "duetsarehowimakefriends",
    "lhrduets",
    "with",
    "withuser",
    "user",
}

ALL_TOPIC_STOPWORDS = TOPIC_MODEL_DOMAIN_STOPWORDS | TOPIC_MODEL_SOCIAL_STOPWORDS
HASH_TOPIC_STOPWORDS = {f"#{word}" for word in ALL_TOPIC_STOPWORDS}


class TikTokAPI:
    def __init__(self, client_key: str, client_secret: str, ratelimit: float = 0.2) -> None:
        if not client_key or not client_secret:
            raise EnvironmentError("TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET must be set.")
        self.client_key = client_key
        self.client_secret = client_secret
        self.base_url = "https://open.tiktokapis.com"
        self.token_url = f"{self.base_url}/v2/oauth/token/"
        self.query_url = f"{self.base_url}/v2/research/video/query/"
        self.comments_query_url = f"{self.base_url}/v2/research/video/comment/list/"
        self.access_token = self.get_access_token()
        self.ratelimit = ratelimit
        self.last_request_time = 0.0

    def get_access_token(self) -> str:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        body = {
            "client_key": self.client_key,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
        }
        response = requests.post(self.token_url, headers=headers, data=body, timeout=60)
        if response.status_code == 200:
            return response.json()["access_token"]
        response.raise_for_status()
        raise RuntimeError("TikTok token request failed without raising an HTTP error.")

    def refresh_token(self) -> None:
        self.access_token = self.get_access_token()

    def query_videos(
        self,
        query_params: dict[str, Any] | None,
        *,
        fields: Sequence[str] = VIDEO_FIELDS,
        start_date: str | None = None,
        end_date: str | None = None,
        is_random: bool = True,
        max_count: int = 100_000,
        count_per_page: int = 100,
        max_trials: int = 3,
        search_id: str | None = None,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> tuple[list[dict[str, Any]], requests.Response | None, Exception | None]:
        payload: dict[str, Any] = {"query": query_params}
        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date
        if count_per_page:
            payload["max_count"] = count_per_page
        if is_random:
            payload["is_random"] = True
        if search_id:
            payload["search_id"] = search_id

        return self._make_paged_request(
            self.query_url,
            payload,
            fields,
            results_entry="videos",
            max_count=max_count,
            max_trials=max_trials,
            verbose=verbose,
            show_progress=show_progress,
        )

    def query_comments(
        self,
        video_id: str,
        *,
        fields: Sequence[str] = COMMENT_FIELDS,
        count_per_page: int = 100,
        max_count: int = 100_000,
        max_trials: int = 3,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> tuple[list[dict[str, Any]], requests.Response | None, Exception | None]:
        payload: dict[str, Any] = {"video_id": video_id}
        if count_per_page:
            payload["max_count"] = count_per_page
        return self._make_paged_request(
            self.comments_query_url,
            payload,
            fields,
            results_entry="comments",
            max_count=max_count,
            max_trials=max_trials,
            verbose=verbose,
            show_progress=show_progress,
        )

    def _sleep_for_rate_limit(self) -> None:
        if self.ratelimit <= 0:
            return
        wait = self.last_request_time + 1.0 / self.ratelimit - time.time()
        if wait > 0:
            time.sleep(wait)

    def _make_paged_request(
        self,
        url: str,
        payload: dict[str, Any],
        fields: Sequence[str] | None,
        *,
        results_entry: str,
        max_count: int,
        max_trials: int,
        verbose: bool,
        show_progress: bool,
    ) -> tuple[list[dict[str, Any]], requests.Response | None, Exception | None]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"bearer {self.access_token}",
        }
        cursor = int(payload.get("cursor", 0) or 0)
        search_id = payload.get("search_id")
        response: requests.Response | None = None
        results: list[dict[str, Any]] = []
        progress = tqdm(total=max_count, desc=results_entry, leave=False) if show_progress else None
        fields_string = f"?fields={','.join(fields)}" if fields else ""

        try:
            while len(results) < max_count:
                request_payload = dict(payload)
                if cursor:
                    request_payload["cursor"] = cursor
                if search_id:
                    request_payload["search_id"] = search_id

                last_error: Exception | None = None
                for trial in range(1, max_trials + 1):
                    try:
                        if verbose:
                            print(f"Starting {results_entry} request trial {trial}/{max_trials}")
                        self._sleep_for_rate_limit()
                        response = requests.post(
                            f"{url}{fields_string}",
                            headers=headers,
                            json=request_payload,
                            timeout=90,
                        )
                        self.last_request_time = time.time()
                        if response.status_code != 200:
                            if verbose:
                                print(response.text)
                            response.raise_for_status()

                        data = response.json().get("data", {})
                        page = data.get(results_entry, []) or []
                        if page:
                            results.extend(page)
                            if progress:
                                progress.update(len(page))
                        has_more = bool(data.get("has_more"))
                        if not has_more or not page:
                            return results[:max_count], response, None

                        cursor = int(data.get("cursor", cursor + len(page)) or 0)
                        search_id = data.get("search_id", search_id)
                        break
                    except Exception as exc:
                        last_error = exc
                        if trial >= max_trials:
                            raise
                        time.sleep(2)
                if last_error and max_trials <= 0:
                    raise last_error
        except Exception as exc:
            return results[:max_count], response, exc
        finally:
            if progress:
                progress.close()

        return results[:max_count], response, None


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_terms(terms: str | None = None, terms_file: Path | None = None) -> list[str]:
    if terms_file:
        loaded = [
            line.strip()
            for line in terms_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    elif terms:
        loaded = parse_csv_list(terms)
    else:
        loaded = MENTAL_HEALTH_TERMS
    deduped: list[str] = []
    seen: set[str] = set()
    for term in loaded:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(term)
    return deduped


def date_window_for_year(year: int, start_mmdd: str, end_mmdd: str) -> tuple[date, date]:
    start = datetime.strptime(f"{year}-{start_mmdd}", "%Y-%m-%d").date()
    end = datetime.strptime(f"{year}-{end_mmdd}", "%Y-%m-%d").date()
    if end < start:
        raise ValueError(f"End date {end} is before start date {start}")
    return start, end


def iter_dates_inclusive(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def api_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def build_research_query(
    terms: Sequence[str],
    *,
    include_keyword: bool = True,
    include_hashtag: bool = True,
    regions: Sequence[str] | None = None,
) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = []
    if include_keyword:
        clauses.extend({"operation": "EQ", "field_name": "keyword", "field_values": [term]} for term in terms)
    if include_hashtag:
        clauses.extend({"operation": "EQ", "field_name": "hashtag_name", "field_values": [term]} for term in terms)
    if not clauses:
        raise ValueError("At least one of include_keyword/include_hashtag must be true.")

    query: dict[str, Any] = {"or": clauses}
    if regions:
        query["and"] = [{"operation": "IN", "field_name": "region_code", "field_values": list(regions)}]
    return query


def write_json_atomically(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(tmp_path, path)


def response_payload(response: requests.Response | None) -> Any:
    if response is None:
        return None
    try:
        return response.json()
    except Exception:
        return response.text


def _string_id(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def coerce_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ID_COLUMNS & set(out.columns):
        out[col] = out[col].map(_string_id).astype("string")
    return out


def read_video_jsons(collected_dir: Path, pattern: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(collected_dir.glob(pattern)):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("results", payload if isinstance(payload, list) else [])
        if not records:
            continue
        frame = pd.DataFrame(records)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return coerce_identifier_columns(pd.concat(frames, ignore_index=True))


def _records_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("results", "comments", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    return []


def read_comment_batches(batch_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    jsonl_files = sorted(batch_dir.glob("*.jsonl"))
    json_files = sorted(batch_dir.glob("*.json"))

    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    records.append(row)

    for path in json_files:
        with path.open("r", encoding="utf-8") as handle:
            records.extend(_records_from_json_payload(json.load(handle)))

    if not records:
        return pd.DataFrame()
    return coerce_identifier_columns(pd.DataFrame(records))


def _parse_hashtags(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set)):
        raw = list(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                raw = parsed if isinstance(parsed, list) else [stripped]
            except Exception:
                raw = [stripped]
        else:
            raw = [part.strip() for part in re.split(r"[, ]+", stripped) if part.strip()]
    else:
        raw = [value]
    tags: list[str] = []
    for item in raw:
        if item is None:
            continue
        tag = str(item).strip().lstrip("#")
        if tag:
            tags.append(tag)
    return tags


def hashtag_text(value: Any) -> str:
    tags = _parse_hashtags(value)
    return " ".join(f"#{tag}" for tag in tags)


def normalize_social_text(text: object) -> str:
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    tokens: list[str] = []
    for token in str(text).split():
        if token.startswith("@") and len(token) > 1:
            tokens.append("@user")
        elif token.startswith("http"):
            tokens.append("http")
        else:
            tokens.append(token)
    return " ".join(tokens)


def compose_video_text(row: pd.Series, *, include_hashtags: bool = True) -> str:
    parts = [
        row.get("video_description", ""),
        row.get("voice_to_text", ""),
    ]
    if include_hashtags:
        parts.append(hashtag_text(row.get("hashtag_names", "")))
    return " ".join(str(part) for part in parts if part is not None and str(part).strip()).strip()


def topic_preprocess_text(text: object) -> str:
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    lowered = str(text).lower()
    cleaned = re.sub(r"[^a-z#\s]", " ", lowered)
    words = cleaned.split()
    words = [
        word.replace("#", "")
        for word in words
        if word not in HASH_TOPIC_STOPWORDS and word.replace("#", "") not in ALL_TOPIC_STOPWORDS
    ]
    return " ".join(words)


def ensure_xlm_polarity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "xlm_polarity" not in out.columns and {"xlm_positive", "xlm_negative"}.issubset(out.columns):
        out["xlm_polarity"] = pd.to_numeric(out["xlm_positive"], errors="coerce") - pd.to_numeric(
            out["xlm_negative"], errors="coerce"
        )
    return out


def model_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def term_tokens(text: str) -> list[str]:
    words = [word for word in str(text).split() if len(word) > 1 and word.isalpha()]
    tokens = list(words)
    tokens.extend("_".join(words[i : i + 2]) for i in range(max(0, len(words) - 1)))
    tokens.extend("_".join(words[i : i + 3]) for i in range(max(0, len(words) - 2)))
    return tokens


def log_odds_keywords(
    df: pd.DataFrame,
    *,
    topic_col: str,
    text_col: str,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = df[pd.to_numeric(df[topic_col], errors="coerce").fillna(-1).astype(int) >= 0].copy()
    if valid.empty:
        empty_terms = pd.DataFrame(columns=["topic_id", "rank", "term", "z_score"])
        empty_map = pd.DataFrame(columns=["topic_id", "topic"])
        return empty_terms, empty_map

    corpora: dict[int, Counter[str]] = defaultdict(Counter)
    background: Counter[str] = Counter()
    for topic_id, text in zip(valid[topic_col], valid[text_col]):
        topic_int = int(topic_id)
        tokens = term_tokens(str(text))
        corpora[topic_int].update(tokens)
        background.update(tokens)

    corpus_sizes = {topic: sum(counter.values()) for topic, counter in corpora.items()}
    background_size = sum(background.values())
    rows: list[dict[str, Any]] = []

    for topic, counter in sorted(corpora.items()):
        other_size = sum(size for key, size in corpus_sizes.items() if key != topic)
        scores: list[tuple[str, float]] = []
        for word, fi in counter.items():
            fj = sum(other_counter[word] for key, other_counter in corpora.items() if key != topic)
            fbg = background[word]
            ni = corpus_sizes[topic]
            nj = other_size
            left_denom = max(1.0, ni + background_size - (fi + fbg))
            right_denom = max(1.0, nj + background_size - (fj + fbg))
            odds_ratio = math.log(fi + fbg) - math.log(left_denom) - math.log(fj + fbg) + math.log(right_denom)
            variance = 1.0 / (fi + fbg) + 1.0 / (fj + fbg)
            z_score = odds_ratio / math.sqrt(variance)
            scores.append((word, z_score))
        for rank, (term, z_score) in enumerate(sorted(scores, key=lambda item: item[1], reverse=True)[:top_n], start=1):
            rows.append({"topic_id": topic, "rank": rank, "term": term, "z_score": z_score})

    terms = pd.DataFrame(rows)
    topic_map = (
        terms.sort_values(["topic_id", "rank"])
        .groupby("topic_id", as_index=False)["term"]
        .agg(lambda values: ", ".join(values))
        .rename(columns={"term": "topic"})
    )
    return terms, topic_map


def write_feather(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_feather(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
