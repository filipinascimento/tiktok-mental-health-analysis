"""Utilities for the TikTok mental health analysis pipeline.

The public API intentionally exposes ``TikTokAPI`` at package level so scripts
can use the same style as the original collection code:

    import tiktokresearch as tiktok
    api = tiktok.TikTokAPI(client_key, client_secret)
"""

from .pipeline import (
    ALL_TOPIC_STOPWORDS,
    COMMENT_FIELDS,
    DATASET_DEFAULT,
    HASH_TOPIC_STOPWORDS,
    MENTAL_HEALTH_TERMS,
    TOPIC_MODEL_DOMAIN_STOPWORDS,
    TOPIC_MODEL_SOCIAL_STOPWORDS,
    VIDEO_FIELDS,
    TikTokAPI,
    api_date,
    build_research_query,
    coerce_identifier_columns,
    compose_video_text,
    date_window_for_year,
    ensure_xlm_polarity,
    iter_dates_inclusive,
    load_terms,
    log_odds_keywords,
    model_device,
    normalize_social_text,
    parse_csv_list,
    read_comment_batches,
    read_video_jsons,
    repo_root_from_script,
    response_payload,
    topic_preprocess_text,
    write_csv,
    write_feather,
    write_json_atomically,
)

__all__ = [
    "ALL_TOPIC_STOPWORDS",
    "COMMENT_FIELDS",
    "DATASET_DEFAULT",
    "HASH_TOPIC_STOPWORDS",
    "MENTAL_HEALTH_TERMS",
    "TOPIC_MODEL_DOMAIN_STOPWORDS",
    "TOPIC_MODEL_SOCIAL_STOPWORDS",
    "VIDEO_FIELDS",
    "TikTokAPI",
    "api_date",
    "build_research_query",
    "coerce_identifier_columns",
    "compose_video_text",
    "date_window_for_year",
    "ensure_xlm_polarity",
    "iter_dates_inclusive",
    "load_terms",
    "log_odds_keywords",
    "model_device",
    "normalize_social_text",
    "parse_csv_list",
    "read_comment_batches",
    "read_video_jsons",
    "repo_root_from_script",
    "response_payload",
    "topic_preprocess_text",
    "write_csv",
    "write_feather",
    "write_json_atomically",
]
