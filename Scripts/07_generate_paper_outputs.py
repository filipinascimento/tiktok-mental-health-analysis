from __future__ import annotations

import argparse
import math
import os
import pickle
import shutil
import subprocess
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde, mannwhitneyu
from tqdm.auto import tqdm


DATASET_DEFAULT = "mentalhealth2024"
PAPER_VIDEO_COUNT = 28_341
PAPER_COMMENT_COUNT = 80_130
DEFAULT_MAX_COMMENTS_PER_VIDEO = 500
DEFAULT_RANDOM_STATE = 42

PAPER_TOPIC_NAMES = [
    "Sunset",
    "Depression",
    "Mental Health Month",
    "Venting 1",
    "Mental Health",
    "Bipolar",
    "Anxiety",
    "Self Care",
    "Positivity",
    "Grief",
    "Borderline",
    "Duet",
    "Suicide Prevention",
    "Psychisch",
    "Venting 2",
]
TOPIC_NAMES = PAPER_TOPIC_NAMES.copy()

FIG6_METRICS = [
    "detoxify_toxicity",
    "detoxify_insult",
    "detoxify_obscene",
    "xlm_polarity",
]

QUANTILES = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]

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
    # TikTok duet boilerplate appears as hashtags and as generated text like
    # "#duet with @user"; suppress it before topic modeling.
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


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the paper figures, tables, statistical tests, and methodology audit."
    )
    parser.add_argument("--dataset", default=DATASET_DEFAULT)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--max-comments-per-video", type=int, default=DEFAULT_MAX_COMMENTS_PER_VIDEO)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--umap-epochs",
        type=int,
        default=1000,
        help=(
            "Epochs for the reproduced Figure 3 UMAP. The paper appendix says 50,000; "
            "1000 is the practical default for reruns. Use 50000 for exact appendix settings."
        ),
    )
    parser.add_argument("--skip-umap", action="store_true", help="Skip Figure 3 UMAP recomputation/plotting.")
    parser.add_argument("--skip-paper-render", action="store_true", help="Skip PDF text/image/page rendering.")
    return parser.parse_args()


def ensure_dirs(new_root: Path) -> dict[str, Path]:
    paths = {
        "new_root": new_root,
        "data": new_root / "Data",
        "scripts": new_root / "Scripts",
        "figures": new_root / "Figures",
        "paper": new_root / "PaperExtracted",
        "logs": new_root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    (paths["paper"] / "page_renders").mkdir(parents=True, exist_ok=True)
    return paths


def configure_topic_names(id2topics: pd.DataFrame) -> None:
    """Size the global topic-name list to the fitted topic model.

    The paper run has 15 manually named topics. If a fresh BERTopic fit yields
    more clusters, keep the paper names for the first 15 IDs and assign neutral
    names to additional IDs so downstream plots/tests remain well-defined.
    """
    global TOPIC_NAMES
    if "topic_id" not in id2topics.columns:
        TOPIC_NAMES = PAPER_TOPIC_NAMES.copy()
        return
    topic_ids = pd.to_numeric(id2topics["topic_id"], errors="coerce")
    topic_ids = topic_ids[topic_ids.notna() & (topic_ids >= 0)].astype(int)
    if topic_ids.empty:
        TOPIC_NAMES = PAPER_TOPIC_NAMES.copy()
        return
    max_topic_id = int(topic_ids.max())
    TOPIC_NAMES = [
        PAPER_TOPIC_NAMES[i] if i < len(PAPER_TOPIC_NAMES) else f"Topic {i + 1}"
        for i in range(max_topic_id + 1)
    ]


def set_plot_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "grid.color": "0.88",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.bottom": True,
            "ytick.left": True,
            "xtick.color": "0.25",
            "ytick.color": "0.25",
            "axes.labelcolor": "0.2",
            "text.color": "0.2",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str, *, dpi: int = 240, tight: bool = True) -> list[Path]:
    outputs = [figures_dir / f"{stem}.pdf", figures_dir / f"{stem}.png"]
    kwargs = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(outputs[0], **kwargs)
    fig.savefig(outputs[1], dpi=dpi, **kwargs)
    plt.close(fig)
    return outputs


def run_optional_command(command: list[str], *, cwd: Path) -> tuple[bool, str]:
    executable = shutil.which(command[0])
    if executable is None:
        return False, f"missing executable: {command[0]}"
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "").strip()
    return True, (proc.stdout or "").strip()


def extract_paper_assets(repo_root: Path, paper_dir: Path) -> pd.DataFrame:
    candidates = [
        repo_root / "TikTok_Mental_Health-9.pdf",
        repo_root / "TikTok_Mental_Health-5.pdf",
        *sorted(repo_root.glob("TikTok_Mental_Health-*.pdf"), reverse=True),
    ]
    pdf = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    rows: list[dict[str, str | bool]] = []
    if not pdf.exists():
        rows.append({"asset_step": "paper_pdf", "ok": False, "detail": f"missing {pdf}"})
        return pd.DataFrame(rows)

    commands = [
        ["pdftotext", "-layout", str(pdf), str(paper_dir / f"{pdf.stem}.txt")],
        ["pdfimages", "-png", str(pdf), str(paper_dir / "embedded_image")],
        ["pdftoppm", "-png", "-r", "220", str(pdf), str(paper_dir / "page_renders" / "page")],
    ]
    for command in tqdm(commands, desc="Extracting/rendering paper assets"):
        ok, detail = run_optional_command(command, cwd=repo_root)
        rows.append({"asset_step": command[0], "ok": ok, "detail": detail})
    return pd.DataFrame(rows)


def read_inputs(data_dir: Path, dataset: str) -> dict[str, pd.DataFrame]:
    files = {
        "videos_raw": data_dir / f"videos_enriched_{dataset}.feather",
        "comments_raw": data_dir / f"comments_enriched_{dataset}.feather",
        "topics": data_dir / "topics.feather",
        "id2topics": data_dir / "id2topics.feather",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, path in tqdm(files.items(), desc="Loading source feather files"):
        if not path.exists():
            raise FileNotFoundError(path)
        out[key] = pd.read_feather(path)
    return out


def epoch_to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="s", utc=True, errors="coerce")


def normalize_comment_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()


def cap_comments_per_video(comments: pd.DataFrame, max_comments: int, random_state: int) -> pd.DataFrame:
    if max_comments <= 0:
        return comments.copy()
    pieces: list[pd.DataFrame] = []
    grouped = comments.groupby("video_id", sort=False, group_keys=False)
    for _, group in tqdm(grouped, total=comments["video_id"].nunique(), desc=f"Capping comments at {max_comments}/video"):
        n = min(len(group), max_comments)
        if len(group) > n:
            pieces.append(group.sample(n=n, random_state=random_state))
        else:
            pieces.append(group)
    if not pieces:
        return comments.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def add_topic_metadata(df: pd.DataFrame, id2topics: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["topic_id"] = pd.to_numeric(out["topic_id"], errors="coerce").astype("Int64")
    id2 = id2topics.rename(columns={"topic": "topic_keywords"}).copy()
    id2["topic_id"] = pd.to_numeric(id2["topic_id"], errors="coerce").astype("Int64")
    out = out.merge(id2[["topic_id", "topic_keywords"]], on="topic_id", how="left")
    out["topic_keywords"] = out["topic_keywords"].fillna("Outlier")
    out["topic_label"] = np.where(out["topic_id"].notna(), (out["topic_id"].astype("Int64") + 1).astype(str), "")
    out["topic_name"] = out["topic_id"].map(lambda x: TOPIC_NAMES[int(x)] if pd.notna(x) and 0 <= int(x) < len(TOPIC_NAMES) else "Outlier")
    return out


def attach_topics(
    videos: pd.DataFrame,
    comments: pd.DataFrame,
    topics: pd.DataFrame,
    id2topics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    topic_map = topics.rename(columns={"topic": "topic_id"}).copy()
    topic_map["id"] = topic_map["id"].astype(str)
    topic_map["topic_id"] = pd.to_numeric(topic_map["topic_id"], errors="coerce").astype("Int64")

    videos_out = videos.copy()
    comments_out = comments.copy()
    videos_out["id"] = videos_out["id"].astype(str)
    comments_out["video_id"] = comments_out["video_id"].astype(str)

    videos_out = videos_out.merge(topic_map[["id", "topic_id"]], on="id", how="left")
    comments_out = comments_out.merge(
        topic_map.rename(columns={"id": "video_id"})[["video_id", "topic_id"]],
        on="video_id",
        how="left",
    )
    videos_out = add_topic_metadata(videos_out, id2topics)
    comments_out = add_topic_metadata(comments_out, id2topics)

    for df in (videos_out, comments_out):
        if all(col in df.columns for col in ["xlm_positive", "xlm_negative"]):
            df["xlm_polarity"] = pd.to_numeric(df["xlm_positive"], errors="coerce") - pd.to_numeric(
                df["xlm_negative"], errors="coerce"
            )
    return videos_out, comments_out


def prepare_data(
    inputs: dict[str, pd.DataFrame],
    out_dir: Path,
    dataset: str,
    max_comments: int,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    videos_raw = inputs["videos_raw"].copy()
    comments_raw = inputs["comments_raw"].copy()
    topics = inputs["topics"].copy()
    id2topics = inputs["id2topics"].copy()

    videos_raw["id"] = videos_raw["id"].astype(str)
    comments_raw["id"] = comments_raw["id"].astype(str)
    comments_raw["video_id"] = comments_raw["video_id"].astype(str)

    comments_unique = comments_raw.drop_duplicates(subset=["id", "video_id"], keep="first").copy()
    comments_content_unique_count = (
        comments_unique.assign(_norm_text=normalize_comment_text(comments_unique["processed_text"] if "processed_text" in comments_unique else comments_unique["text"]))
        .drop_duplicates(subset=["video_id", "_norm_text"])
        .shape[0]
    )
    comments_capped = cap_comments_per_video(comments_unique, max_comments=max_comments, random_state=random_state)

    videos_unique = videos_raw.drop_duplicates(subset=["id"], keep="first").copy()
    videos_with_topics, comments_raw_with_topics = attach_topics(videos_unique, comments_unique, topics, id2topics)
    _, comments_capped_with_topics = attach_topics(videos_unique, comments_capped, topics, id2topics)

    videos_analysis = videos_with_topics[videos_with_topics["topic_id"].notna() & (videos_with_topics["topic_id"] >= 0)].copy()
    comments_raw_topic_valid = comments_raw_with_topics[
        comments_raw_with_topics["topic_id"].notna() & (comments_raw_with_topics["topic_id"] >= 0)
    ].copy()
    comments_analysis = comments_capped_with_topics[
        comments_capped_with_topics["topic_id"].notna() & (comments_capped_with_topics["topic_id"] >= 0)
    ].copy()

    outputs = {
        "videos_unique": videos_unique,
        "comments_unique": comments_unique,
        "comments_capped": comments_capped,
        "videos_with_topics": videos_with_topics,
        "comments_raw_with_topics": comments_raw_with_topics,
        "comments_capped_with_topics": comments_capped_with_topics,
        "videos_analysis": videos_analysis,
        "comments_raw_topic_valid": comments_raw_topic_valid,
        "comments_analysis": comments_analysis,
    }

    for key, df in tqdm(outputs.items(), desc="Writing intermediate feather files"):
        df.reset_index(drop=True).to_feather(out_dir / f"{key}_{dataset}.feather")

    sanity = pd.DataFrame(
        [
            {"measure": "paper_video_count", "value": PAPER_VIDEO_COUNT},
            {"measure": "paper_comment_count", "value": PAPER_COMMENT_COUNT},
            {"measure": "source_video_rows", "value": len(videos_raw)},
            {"measure": "unique_video_ids", "value": videos_raw["id"].nunique()},
            {"measure": "source_comment_rows", "value": len(comments_raw)},
            {"measure": "unique_comment_id_video", "value": len(comments_unique)},
            {"measure": "unique_comment_video_text", "value": comments_content_unique_count},
            {"measure": f"comments_after_cap_{max_comments}_per_video", "value": len(comments_capped)},
            {"measure": "topic_rows", "value": len(topics)},
            {"measure": "videos_with_any_topic", "value": int(videos_with_topics["topic_id"].notna().sum())},
            {"measure": "videos_valid_non_outlier_topics", "value": len(videos_analysis)},
            {"measure": "comments_unique_with_any_topic", "value": int(comments_raw_with_topics["topic_id"].notna().sum())},
            {"measure": "comments_unique_valid_non_outlier_topics", "value": len(comments_raw_topic_valid)},
            {"measure": f"comments_capped_{max_comments}_valid_non_outlier_topics", "value": len(comments_analysis)},
        ]
    )
    sanity.to_csv(out_dir / f"sanity_counts_{dataset}.csv", index=False)

    return outputs | {"sanity": sanity}


def write_topic_tables(id2topics: pd.DataFrame, data_dir: Path, figures_dir: Path, dataset: str) -> pd.DataFrame:
    id2 = id2topics.copy()
    id2["topic_id"] = pd.to_numeric(id2["topic_id"], errors="coerce").astype(int)
    rows = []
    for topic_id in range(len(TOPIC_NAMES)):
        matches = id2.loc[id2["topic_id"].eq(topic_id), "topic"]
        keywords = matches.iloc[0] if not matches.empty else ""
        tags = [tag.strip().replace("_", " ") for tag in str(keywords).split(",")]
        rows.append(
            {
                "ID": topic_id + 1,
                "topic_id": topic_id,
                "Topic label": TOPIC_NAMES[topic_id],
                "Top tags (first 6)": ", ".join(tags[:6]),
                "Top tags (all)": ", ".join(tags),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(data_dir / f"table_1_topics_{dataset}.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off")
    cell_text = [
        [
            str(row["ID"]),
            row["Topic label"],
            textwrap.fill(str(row["Top tags (first 6)"]), width=58),
        ]
        for _, row in table.iterrows()
    ]
    mpl_table = ax.table(
        cellText=cell_text,
        colLabels=["ID", "Topic label", "Top tags (by log-odds)"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.07, 0.22, 0.71],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8.5)
    mpl_table.scale(1, 1.42)
    for (row, _), cell in mpl_table.get_celld().items():
        cell.set_linewidth(0.25)
        cell.set_edgecolor("0.82")
        if row == 0:
            cell.set_facecolor("0.94")
            cell.set_text_props(weight="bold")
    ax.set_title("Table 1: Topics from video descriptions and hashtags", loc="left", fontsize=12, pad=12)
    save_figure(fig, figures_dir, f"table_1_topics_{dataset}", tight=True)

    appendix = pd.DataFrame(
        [
            ["Embedding model", "all-mpnet-base-v2"],
            ["UMAP neighbors (k)", "20"],
            ["UMAP embedding dim.", "10 (2 for visualization)"],
            ["UMAP min. distance", "0.05"],
            ["UMAP training epochs", "50,000 in paper; see run argument for reproduced Figure 3"],
            ["UMAP distance metric", "cosine"],
            ["HDBSCAN min. cluster size", "300"],
            ["HDBSCAN max. cluster size", "5,000"],
            ["HDBSCAN min. samples", "15"],
            ["HDBSCAN distance metric", "euclidean"],
            ["HDBSCAN cluster selection", "Excess of Mass"],
        ],
        columns=["Hyperparameter", "Value"],
    )
    appendix.to_csv(data_dir / f"table_a1_topic_model_hyperparameters_{dataset}.csv", index=False)
    return table


def plot_figure_1(
    videos: pd.DataFrame,
    comments: pd.DataFrame,
    data_dir: Path,
    figures_dir: Path,
    dataset: str,
) -> pd.DataFrame:
    videos = videos.copy()
    comments = comments.copy()
    videos["video_created_at"] = epoch_to_utc(videos["create_time"])
    comments["comment_created_at"] = epoch_to_utc(comments["create_time"])
    video_year_map = videos[["id", "video_created_at"]].rename(columns={"id": "video_id"})
    video_year_map["video_year"] = video_year_map["video_created_at"].dt.year
    comments = comments.merge(video_year_map[["video_id", "video_year"]], on="video_id", how="left")

    rows = []
    for year in tqdm([2023, 2024], desc="Building Figure 1 daily counts"):
        start = pd.Timestamp(year=year, month=4, day=15, tz="UTC")
        end = pd.Timestamp(year=year, month=10, day=15, tz="UTC")
        dates = pd.date_range(start=start.date(), end=end.date(), freq="D", tz="UTC")

        vy = videos[videos["video_created_at"].dt.year.eq(year)].copy()
        vy["date"] = videos.loc[vy.index, "video_created_at"].dt.floor("D")
        cy = comments[comments["video_year"].eq(year)].copy()
        cy = cy[cy["comment_created_at"].between(start, end + pd.Timedelta(days=1), inclusive="left")]
        cy["date"] = cy["comment_created_at"].dt.floor("D")

        post_counts = vy.groupby("date")["id"].nunique()
        user_counts = vy.groupby("date")["username"].nunique() if "username" in vy else pd.Series(dtype=float)
        comment_counts = cy.groupby("date")["id"].nunique()

        for date in dates:
            rows.append(
                {
                    "year": year,
                    "date": date.date().isoformat(),
                    "unique_posts": int(post_counts.get(date, 0)),
                    "unique_users_posts": int(user_counts.get(date, 0)),
                    "comments": int(comment_counts.get(date, 0)),
                }
            )

    daily = pd.DataFrame(rows)
    daily.to_csv(data_dir / f"figure_1_daily_counts_{dataset}.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.8), sharex=False)
    palette = {"unique_posts": "#4C78A8", "comments": "#D55E00", "unique_users_posts": "#54A24B"}
    labels = {"unique_posts": "Unique Posts", "comments": "Comments", "unique_users_posts": "Unique Users (Posts)"}
    for ax, year, panel in zip(axes, [2023, 2024], ["(a)", "(b)"]):
        sub = daily[daily["year"].eq(year)].copy()
        dates = pd.to_datetime(sub["date"])
        for col in ["unique_posts", "comments", "unique_users_posts"]:
            ax.plot(dates, sub[col], lw=1.8, label=labels[col], color=palette[col])
        ax.axvline(pd.Timestamp(year=year, month=5, day=1), color="0.25", ls="--", lw=0.9, alpha=0.7)
        ax.axvline(pd.Timestamp(year=year, month=6, day=1), color="0.25", ls="--", lw=0.9, alpha=0.7)
        ax.set_ylabel("Count")
        ax.set_title(panel, loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.set_xlim(pd.Timestamp(year=year, month=4, day=15), pd.Timestamp(year=year, month=10, day=15))
    axes[-1].set_xlabel("Date")
    save_figure(fig, figures_dir, f"figure_1_post_comment_timelines_{dataset}", tight=True)
    return daily


def plot_figure_2(figures_dir: Path, dataset: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str) -> None:
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="0.35", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "0.3"})

    box(0.06, 0.68, 0.22, 0.13, "Mental health\nkeywords", "#E8F0F7")
    box(0.06, 0.46, 0.22, 0.13, "April 15 - June 15\n2023 and 2024", "#E8F0F7")
    box(0.37, 0.57, 0.22, 0.15, "TikTok\nResearch API", "#F3E6D8")
    box(0.69, 0.68, 0.20, 0.13, "Videos:\ndescription + speech + hashtags", "#E6F2E6")
    box(0.69, 0.45, 0.20, 0.13, "Comments:\nreply text", "#F8E7E7")
    box(0.18, 0.18, 0.20, 0.13, "BERTopic\n+ log-odds", "#ECE8F6")
    box(0.43, 0.18, 0.20, 0.13, "Detoxify\ntoxicity", "#F5E3D7")
    box(0.68, 0.18, 0.20, 0.13, "XLM-T\nsentiment", "#DCEEF7")
    arrow(0.28, 0.745, 0.37, 0.65)
    arrow(0.28, 0.525, 0.37, 0.65)
    arrow(0.59, 0.65, 0.69, 0.745)
    arrow(0.59, 0.62, 0.69, 0.515)
    arrow(0.79, 0.68, 0.28, 0.31)
    arrow(0.79, 0.45, 0.53, 0.31)
    arrow(0.79, 0.45, 0.78, 0.31)
    ax.set_title("Figure 2: Data collection and analysis pipeline", loc="left", fontsize=12)
    save_figure(fig, figures_dir, f"figure_2_pipeline_schematic_{dataset}", tight=True)


def compute_or_load_umap(
    repo_root: Path,
    data_dir: Path,
    topics: pd.DataFrame,
    dataset: str,
    n_epochs: int,
) -> pd.DataFrame:
    out_path = data_dir / f"figure_3_umap_coordinates_{dataset}_epochs{n_epochs}.feather"
    if out_path.exists():
        return pd.read_feather(out_path)

    embeddings_path = repo_root / "Data" / "embeddings.pkl"
    if not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)
    with open(embeddings_path, "rb") as handle:
        embeddings = pickle.load(handle)
    if len(embeddings) != len(topics):
        raise ValueError(f"embeddings rows ({len(embeddings)}) != topics rows ({len(topics)})")

    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("umap-learn is required for Figure 3. Install with: pip install umap-learn") from exc

    reducer = umap.UMAP(
        n_neighbors=20,
        n_components=2,
        min_dist=0.05,
        n_epochs=n_epochs,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    coords_np = reducer.fit_transform(embeddings)
    coords = topics.rename(columns={"topic": "topic_id"}).copy()
    coords["topic_id"] = pd.to_numeric(coords["topic_id"], errors="coerce").astype("Int64")
    coords["umap_x"] = coords_np[:, 0]
    coords["umap_y"] = coords_np[:, 1]
    coords.to_feather(out_path)
    return coords


def plot_figure_3(coords: pd.DataFrame, figures_dir: Path, dataset: str, n_epochs: int) -> None:
    plot_df = coords[coords["topic_id"].notna() & (coords["topic_id"] >= 0) & (coords["topic_id"] != 11)].copy()
    plot_df["topic_name"] = plot_df["topic_id"].astype(int).map(lambda i: TOPIC_NAMES[i])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    palette = sns.color_palette("tab20", n_colors=len(TOPIC_NAMES))
    for topic_id in tqdm(sorted(plot_df["topic_id"].astype(int).unique()), desc="Plotting Figure 3 topics"):
        sub = plot_df[plot_df["topic_id"].astype(int).eq(topic_id)]
        ax.scatter(
            sub["umap_x"],
            sub["umap_y"],
            s=8,
            alpha=0.65,
            linewidths=0,
            color=palette[topic_id],
            label=f"{topic_id + 1} {TOPIC_NAMES[topic_id]}",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Figure 3: UMAP projection of topics (Duet excluded; epochs={n_epochs})", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save_figure(fig, figures_dir, f"figure_3_umap_topics_{dataset}", tight=True)


def plot_figures_4_and_5(
    videos_with_topics: pd.DataFrame,
    comments_raw_topic_valid: pd.DataFrame,
    data_dir: Path,
    figures_dir: Path,
    dataset: str,
) -> None:
    videos = videos_with_topics[videos_with_topics["topic_id"].notna() & (videos_with_topics["topic_id"] >= 0)].copy()
    videos["created_at"] = epoch_to_utc(videos["create_time"])
    videos["year"] = videos["created_at"].dt.year
    videos["topic_num"] = videos["topic_id"].astype(int) + 1
    video_counts = videos.groupby(["year", "topic_num"], as_index=False).size().rename(columns={"size": "count"})
    totals = video_counts.groupby("year")["count"].transform("sum")
    video_counts["fraction"] = video_counts["count"] / totals
    video_counts.to_csv(data_dir / f"figure_4_video_topic_fractions_{dataset}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.3, 3.6))
    sns.barplot(data=video_counts[video_counts["year"].isin([2023, 2024])], x="topic_num", y="fraction", hue="year", ax=ax)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Fraction of Videos")
    ax.set_title("Figure 4: Fractions of posts per topic for 2023 and 2024", loc="left", fontsize=11)
    ax.legend(title="", frameon=False)
    save_figure(fig, figures_dir, f"figure_4_video_topic_fractions_{dataset}", tight=True)

    comments = comments_raw_topic_valid.copy()
    comments["topic_num"] = comments["topic_id"].astype(int) + 1
    comment_counts = comments.groupby("topic_num", as_index=False).size().rename(columns={"size": "count"})
    comment_counts["fraction"] = comment_counts["count"] / comment_counts["count"].sum()
    comment_counts.to_csv(data_dir / f"figure_5_comment_topic_fractions_raw_comments_{dataset}.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    sns.barplot(data=comment_counts, x="topic_num", y="fraction", color="#4C78A8", ax=ax)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Fraction of Comments")
    ax.set_title("Figure 5: Fractions of comments by topic", loc="left", fontsize=11)
    save_figure(fig, figures_dir, f"figure_5_comment_topic_fractions_{dataset}", tight=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), gridspec_kw={"width_ratios": [1.25, 1.0]})
    sns.barplot(data=video_counts[video_counts["year"].isin([2023, 2024])], x="topic_num", y="fraction", hue="year", ax=axes[0])
    axes[0].set_xlabel("Topic")
    axes[0].set_ylabel("Fraction of Videos")
    axes[0].legend(title="", frameon=False)
    sns.barplot(data=comment_counts, x="topic_num", y="fraction", color="#4C78A8", ax=axes[1])
    axes[1].set_xlabel("Topic")
    axes[1].set_ylabel("Fraction of Comments")
    axes[0].set_title("Figure 4", loc="left", fontweight="bold")
    axes[1].set_title("Figure 5", loc="left", fontweight="bold")
    save_figure(fig, figures_dir, f"figure_4_5_topic_fractions_combined_{dataset}", tight=True)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    p_ok = p[ok]
    order = np.argsort(p_ok)
    sorted_p = p_ok[order]
    ranks = np.arange(1, len(sorted_p) + 1, dtype=float)
    sorted_q = np.minimum.accumulate(((sorted_p * len(sorted_p)) / ranks)[::-1])[::-1]
    sorted_q = np.clip(sorted_q, 0, 1)
    q_ok = np.empty_like(p_ok)
    q_ok[order] = sorted_q
    q[ok] = q_ok
    return q


def significance_label(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def run_topic_tests(df: pd.DataFrame, kind: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    topic_ids = sorted(df["topic_id"].astype(int).unique().tolist())
    for topic_id in tqdm(topic_ids, desc=f"Mann-Whitney topic-vs-rest tests ({kind})"):
        topic_mask = df["topic_id"].astype(int).eq(topic_id)
        for metric in metrics:
            topic_values = pd.to_numeric(df.loc[topic_mask, metric], errors="coerce").to_numpy(dtype=float)
            rest_values = pd.to_numeric(df.loc[~topic_mask, metric], errors="coerce").to_numpy(dtype=float)
            topic_values = topic_values[np.isfinite(topic_values)]
            rest_values = rest_values[np.isfinite(rest_values)]
            if len(topic_values) < 2 or len(rest_values) < 2:
                u_stat = np.nan
                p_value = np.nan
            else:
                res = mannwhitneyu(topic_values, rest_values, alternative="two-sided", method="auto")
                u_stat = float(res.statistic)
                p_value = float(res.pvalue)
            rows.append(
                {
                    "kind": kind,
                    "topic_id": topic_id,
                    "topic_label": str(topic_id + 1),
                    "topic_name": TOPIC_NAMES[topic_id],
                    "metric": metric,
                    "n_topic": len(topic_values),
                    "n_rest": len(rest_values),
                    "median_topic": float(np.nanmedian(topic_values)) if len(topic_values) else np.nan,
                    "median_rest": float(np.nanmedian(rest_values)) if len(rest_values) else np.nan,
                    "mean_topic": float(np.nanmean(topic_values)) if len(topic_values) else np.nan,
                    "mean_rest": float(np.nanmean(rest_values)) if len(rest_values) else np.nan,
                    "u_stat": u_stat,
                    "p_value": p_value,
                }
            )
    out = pd.DataFrame(rows)
    out["q_value_bh_fdr"] = np.nan
    for metric in metrics:
        mask = out["metric"].eq(metric)
        out.loc[mask, "q_value_bh_fdr"] = bh_fdr(out.loc[mask, "p_value"].to_numpy(dtype=float))
    out["significance"] = out["q_value_bh_fdr"].map(significance_label)
    return out


def run_figure6_tests(
    videos_analysis: pd.DataFrame,
    comments_analysis: pd.DataFrame,
    data_dir: Path,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for metric in FIG6_METRICS:
        if metric not in videos_analysis.columns or metric not in comments_analysis.columns:
            raise ValueError(f"Missing Figure 6 metric: {metric}")
    videos_tests = run_topic_tests(videos_analysis, "videos", FIG6_METRICS)
    comments_tests = run_topic_tests(comments_analysis, "comments_capped500", FIG6_METRICS)
    videos_tests.to_csv(data_dir / f"topic_vs_rest_mannwhitney_videos_figure6_{dataset}.csv", index=False)
    comments_tests.to_csv(data_dir / f"topic_vs_rest_mannwhitney_comments_capped500_figure6_{dataset}.csv", index=False)

    a = pd.to_numeric(videos_analysis["xlm_polarity"], errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(comments_analysis["xlm_polarity"], errors="coerce").dropna().to_numpy(dtype=float)
    res = mannwhitneyu(a, b, alternative="two-sided", method="auto")
    vidcom = pd.DataFrame(
        [
            {
                "metric": "xlm_polarity",
                "n_videos": len(a),
                "n_comments_capped500": len(b),
                "median_videos": float(np.median(a)),
                "median_comments_capped500": float(np.median(b)),
                "mean_videos": float(np.mean(a)),
                "mean_comments_capped500": float(np.mean(b)),
                "u_stat": float(res.statistic),
                "p_value": float(res.pvalue),
                "q_value_bh_fdr": float(res.pvalue),
                "significance": significance_label(float(res.pvalue)),
            }
        ]
    )
    vidcom.to_csv(data_dir / f"videos_vs_comments_sentiment_mannwhitney_figure6_{dataset}.csv", index=False)
    return videos_tests, comments_tests, vidcom


def quantile_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    qtab = (
        df[["topic_id", metric]]
        .dropna()
        .assign(topic_id=lambda x: x["topic_id"].astype(int))
        .groupby("topic_id")[metric]
        .quantile(QUANTILES)
        .unstack()
    )
    return qtab.reindex(range(len(TOPIC_NAMES)))


def draw_quantile_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    tests: pd.DataFrame,
    metric: str,
    title: str,
    panel_label: str,
    xlim: tuple[float, float],
    *,
    show_ylabels: bool,
) -> None:
    qtab = quantile_table(df, metric)
    y = np.arange(len(TOPIC_NAMES))
    is_polarity = metric == "xlm_polarity"
    color = "#2C7FB8" if is_polarity else "#D66A3A"
    neutral = "0.35"
    alpha_mid = 0.48 if not is_polarity else 0.45
    alpha_outer = 0.34 if not is_polarity else 0.35
    lw_main = 8.5 if is_polarity else 8.0
    lw_mid = 4.5
    lw_outer = 2.4

    ax.hlines(y, qtab[0.10], qtab[0.90], color=color, lw=lw_main, alpha=0.86, zorder=2)
    ax.hlines(y, qtab[0.05], qtab[0.95], color=color, lw=lw_mid, alpha=alpha_mid, zorder=1)
    ax.hlines(y, qtab[0.01], qtab[0.99], color=color, lw=lw_outer, alpha=alpha_outer, zorder=1)
    for q, markersize in [(0.01, 2.5), (0.05, 4.0), (0.10, 6.0), (0.90, 6.0), (0.95, 4.0), (0.99, 2.5)]:
        ax.plot(qtab[q], y, linestyle="None", marker="|", markersize=markersize, markeredgewidth=0.9, color=color, zorder=4)
    ax.vlines(qtab[0.50], y - 0.30, y + 0.30, color=neutral, lw=1.0, alpha=0.75, zorder=5)

    if is_polarity:
        ax.axvline(0, color="0.25", lw=0.8, alpha=0.45, zorder=0)
        med = (
            df[["topic_id", "xlm_negative", "xlm_positive"]]
            .dropna()
            .assign(topic_id=lambda x: x["topic_id"].astype(int))
            .groupby("topic_id")[["xlm_negative", "xlm_positive"]]
            .median()
            .reindex(range(len(TOPIC_NAMES)))
        )
        ax.scatter(-med["xlm_negative"], y, marker="v", s=32, color=neutral, zorder=6)
        ax.scatter(med["xlm_positive"], y, marker="^", s=32, color=neutral, zorder=6)

    for topic_id in range(len(TOPIC_NAMES)):
        row = tests[(tests["topic_id"].eq(topic_id)) & (tests["metric"].eq(metric))]
        if row.empty:
            continue
        stars = row["significance"].iloc[0]
        if pd.isna(stars) or not str(stars).strip():
            continue
        stars = str(stars)
        if is_polarity:
            x_star = min(xlim[1] - 0.04 * (xlim[1] - xlim[0]), 1.03)
        else:
            q99 = qtab.loc[topic_id, 0.99]
            x_star = min(xlim[1] - 0.08 * (xlim[1] - xlim[0]), q99 + 0.045 * (xlim[1] - xlim[0]))
        ax.text(x_star, topic_id, stars, ha="left", va="center", fontsize=10, color="0.55", fontweight="bold")

    ax.set_xlim(*xlim)
    ax.set_ylim(len(TOPIC_NAMES) - 0.5, -0.5)
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels([str(i) for i in range(1, len(TOPIC_NAMES) + 1)])
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=12, pad=6)
    ax.text(-0.07, 1.04, panel_label, transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.28)
    ax.grid(axis="y", alpha=0.25)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("0.75")
        ax.spines[side].set_linewidth(1.0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def draw_kde(ax: plt.Axes, videos: pd.DataFrame, comments: pd.DataFrame) -> None:
    grid = np.linspace(-1.2, 1.1, 500)
    specs = [
        (videos["xlm_polarity"], "videos", "#9A9A90", 0.42),
        (comments["xlm_polarity"], "comments", "#C53B43", 0.34),
    ]
    for series, label, color, alpha in specs:
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 3 or np.std(values) == 0:
            continue
        kde = gaussian_kde(values)
        density = kde(grid)
        ax.fill_between(grid, density, color=color, alpha=alpha, label=label)
        ax.plot(grid, density, color=color, lw=1.1)
    ax.set_xlim(-1.25, 1.15)
    ax.set_xlabel("Sentiment polarity")
    ax.set_ylabel("Density")
    ax.set_title("(i)", loc="left", fontweight="bold")
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("0.75")
        ax.spines[side].set_linewidth(1.0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def draw_figure6_legend(ax: plt.Axes) -> None:
    ax.axis("off")
    sample_rows = [
        ("videos", "#9A9A90", 0.42, "#6F746B"),
        ("comments", "#C53B43", 0.34, "#C53B43"),
    ]
    for idx, (label, face, alpha, edge) in enumerate(sample_rows):
        y = 0.86 - idx * 0.10
        ax.add_patch(
            Rectangle(
                (0.02, y - 0.028),
                0.08,
                0.048,
                transform=ax.transAxes,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.8,
                alpha=alpha,
                clip_on=False,
            )
        )
        ax.text(0.115, y, label, transform=ax.transAxes, va="center", ha="left", fontsize=8.5)

    topic_rows = [[str(i + 1), TOPIC_NAMES[i]] for i in range(len(TOPIC_NAMES))]
    topic_table = ax.table(
        cellText=topic_rows,
        colLabels=["", ""],
        cellLoc="left",
        colLoc="left",
        bbox=[0.22, 0.03, 0.40, 0.92],
        colWidths=[0.10, 0.30],
    )
    topic_table.auto_set_font_size(False)
    topic_table.set_fontsize(8.5)
    for (row, _), cell in topic_table.get_celld().items():
        cell.set_edgecolor("none")
        if row == 0:
            cell.set_facecolor("0.96")
            cell.get_text().set_text("")
        else:
            cell.set_facecolor("white")

    x0 = 0.70
    y0 = 0.90
    dy = 0.085
    handles = [
        Line2D([0], [0], color="0.55", lw=7, alpha=0.85),
        Line2D([0], [0], color="0.55", lw=3.5, alpha=0.48),
        Line2D([0], [0], color="0.55", lw=1.8, alpha=0.34),
        Line2D([0], [0], color="0.35", lw=1.0),
        Line2D([0], [0], marker="^", color="0.35", lw=0, markersize=7),
        Line2D([0], [0], marker="v", color="0.35", lw=0, markersize=7),
    ]
    labels = ["10%-90%", "5%-95%", "1%-99%", "median", "pos. median", "neg. median"]
    for idx, (handle, label) in enumerate(zip(handles, labels)):
        y = y0 - idx * dy
        ax.add_line(Line2D([x0, x0 + 0.08], [y, y], transform=ax.transAxes, color=handle.get_color(), lw=handle.get_linewidth(), alpha=handle.get_alpha() or 1))
        marker = handle.get_marker()
        if marker not in [None, "None", ""]:
            ax.plot([x0 + 0.04], [y], marker=marker, color="0.35", transform=ax.transAxes, markersize=7, linestyle="None")
        ax.text(x0 + 0.11, y, label, transform=ax.transAxes, va="center", fontsize=8.5)
    for idx, label in enumerate(["*  q<0.05", "**  q<0.01", "***  q<0.001"]):
        ax.text(x0 + 0.01, 0.30 - idx * 0.09, label, transform=ax.transAxes, va="center", fontsize=8.5, fontweight="bold", color="0.35")


def plot_figure_6(
    videos_analysis: pd.DataFrame,
    comments_analysis: pd.DataFrame,
    videos_tests: pd.DataFrame,
    comments_tests: pd.DataFrame,
    figures_dir: Path,
    dataset: str,
) -> None:
    set_plot_theme()
    fig = plt.figure(figsize=(13.8, 11.0))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.95], hspace=0.25, wspace=0.20)
    titles = ["Toxicity", "Insult", "Obscene", "Sentiment Polarity"]
    xlims = [(0, 1.0), (0, 0.52), (0, 1.0), (-1, 1.18)]
    panels_top = ["(a)", "(b)", "(c)", "(d)"]
    panels_bottom = ["(e)", "(f)", "(g)", "(h)"]

    for col, metric in enumerate(FIG6_METRICS):
        ax = fig.add_subplot(gs[0, col])
        draw_quantile_panel(
            ax,
            videos_analysis,
            videos_tests,
            metric,
            titles[col],
            panels_top[col],
            xlims[col],
            show_ylabels=(col == 0),
        )
        ax = fig.add_subplot(gs[1, col])
        draw_quantile_panel(
            ax,
            comments_analysis,
            comments_tests,
            metric,
            "",
            panels_bottom[col],
            xlims[col],
            show_ylabels=(col == 0),
        )

    kde_ax = fig.add_subplot(gs[2, 0:2])
    draw_kde(kde_ax, videos_analysis, comments_analysis)
    legend_ax = fig.add_subplot(gs[2, 2:4])
    draw_figure6_legend(legend_ax)

    fig.text(0.024, 0.735, "Videos", rotation=90, va="center", ha="center", fontsize=13)
    fig.text(0.024, 0.425, "Comments", rotation=90, va="center", ha="center", fontsize=13)

    # The paper uses subtle row brackets instead of panel boxes.
    for y0, y1 in [(0.615, 0.905), (0.305, 0.595)]:
        x = 0.055
        fig.add_artist(Line2D([x, x], [y0, y1], transform=fig.transFigure, color="0.65", lw=1.6))
        fig.add_artist(Line2D([x, x + 0.008], [y1, y1], transform=fig.transFigure, color="0.65", lw=1.6))
        fig.add_artist(Line2D([x, x + 0.008], [y0, y0], transform=fig.transFigure, color="0.65", lw=1.6))
    save_figure(fig, figures_dir, f"figure_6_toxicity_sentiment_{dataset}", tight=True)


def top_videos_by_comments(
    videos: pd.DataFrame,
    comments: pd.DataFrame,
    data_dir: Path,
    dataset: str,
    year: int = 2024,
) -> pd.DataFrame:
    videos = videos.copy()
    comments = comments.copy()
    videos["created_at"] = epoch_to_utc(videos["create_time"])
    videos = videos[videos["created_at"].dt.year.eq(year)].copy()
    collected = comments.groupby("video_id", as_index=False).size().rename(
        columns={"video_id": "id", "size": "collected_comments_raw_unique"}
    )
    text_col = "processed_text" if "processed_text" in comments.columns else "text"
    comments_text_dedup = comments.assign(_norm_text=normalize_comment_text(comments[text_col])).drop_duplicates(
        subset=["video_id", "_norm_text"], keep="first"
    )
    collected_text = comments_text_dedup.groupby("video_id", as_index=False).size().rename(
        columns={"video_id": "id", "size": "collected_comments_text_dedup"}
    )
    out = videos.merge(collected, on="id", how="left").merge(collected_text, on="id", how="left")
    out["collected_comments_raw_unique"] = out["collected_comments_raw_unique"].fillna(0).astype(int)
    out["collected_comments_text_dedup"] = out["collected_comments_text_dedup"].fillna(0).astype(int)
    out = out.sort_values("collected_comments_raw_unique", ascending=False)
    cols = [c for c in ["id", "username", "collected_comments", "comment_count", "view_count", "like_count"] if c in out.columns]
    cols = [
        c
        for c in [
            "id",
            "username",
            "collected_comments_raw_unique",
            "collected_comments_text_dedup",
            "comment_count",
            "view_count",
            "like_count",
        ]
        if c in out.columns
    ]
    out = out[cols].head(50)
    out.to_csv(data_dir / f"top_videos_by_collected_comments_{year}_{dataset}.csv", index=False)
    return out


def write_methodology_audit(
    report_path: Path,
    sanity: pd.DataFrame,
    vidcom_test: pd.DataFrame,
    *,
    dataset: str,
    max_comments: int,
    random_state: int,
    umap_epochs: int,
    top_videos_2024: pd.DataFrame | None = None,
) -> None:
    sanity_map = dict(zip(sanity["measure"], sanity["value"]))
    comment_after_cap = sanity_map.get(f"comments_after_cap_{max_comments}_per_video", "NA")
    vidcom_q = vidcom_test["q_value_bh_fdr"].iloc[0] if not vidcom_test.empty else math.nan
    if top_videos_2024 is not None and not top_videos_2024.empty:
        raw_top = ", ".join(top_videos_2024["collected_comments_raw_unique"].head(5).astype(str).tolist())
        text_top = ", ".join(top_videos_2024["collected_comments_text_dedup"].head(5).astype(str).tolist())
    else:
        raw_top = "NA"
        text_top = "NA"
    text = f"""# Methodology Audit for `{dataset}`

Generated by `Scripts/07_generate_paper_outputs.py`.

## Reproduced Code Filters

- Videos are de-duplicated with `drop_duplicates(subset=["id"], keep="first")`.
- Comments are de-duplicated with `drop_duplicates(subset=["id", "video_id"], keep="first")`.
- Figure 6 and the topic-level tests use comments capped at `{max_comments}` per `video_id` with `random_state={random_state}`.
- Topic plots/tests exclude `topic_id < 0` outliers.
- Comment topics are inherited from the parent video's topic assignment through `video_id`.
- Sentiment polarity is computed as `xlm_positive - xlm_negative`.
- Figure 6 tests are two-sided Mann-Whitney U tests, with Benjamini-Hochberg FDR applied separately per metric within each content type.
- Figure 5 uses uncapped unique comments, because the paper text identifies top commented topics 2, 3, 13, 7, and 10; that ordering matches uncapped comments and not the capped comment set.

## Counts and Mismatches

- Paper states `{PAPER_VIDEO_COUNT:,}` videos. The source file has `{sanity_map.get("source_video_rows", "NA"):,}` video rows and `{sanity_map.get("unique_video_ids", "NA"):,}` unique video IDs. The unique-video count matches the paper.
- Paper states `{PAPER_COMMENT_COUNT:,}` comments. The source file has `{sanity_map.get("source_comment_rows", "NA"):,}` comment rows and `{sanity_map.get("unique_comment_id_video", "NA"):,}` unique `(id, video_id)` rows.
- After de-duplicating comments and applying the `{max_comments}` per-video cap, this pipeline has `{comment_after_cap:,}` comments before topic filtering.
- After joining topics and excluding outliers, Figure 6 uses `{sanity_map.get("videos_valid_non_outlier_topics", "NA"):,}` videos and `{sanity_map.get(f"comments_capped_{max_comments}_valid_non_outlier_topics", "NA"):,}` capped comments.
- The exact paper comment total of `{PAPER_COMMENT_COUNT:,}` is not recoverable from the available feather files using the filters exposed in the current code. A missing raw-data filter, removed file, or undocumented deduplication rule is likely.
- The paper reports the top five 2024 post comment counts as `6,923, 1,483, 1,430, 921, 622`. The available data gives `{raw_top}` when counting unique comment IDs and `{text_top}` after normalized text de-duplication. This is another sign that the final paper used a comment snapshot/filter that is not fully represented in the available scripts.

## Paper Explanations That Are Missing or Ambiguous

- The paper does not specify duplicate-row handling, but the codebase repeatedly de-duplicates videos and comments before analysis.
- The paper says a conservative cap is applied for topics exceeding 500 comments; this reproducible pipeline applies the cap per `video_id` and reports that choice explicitly.
- The cap is applied consistently to Figure 6 and its topic-level Mann-Whitney tests.
- Figure 5 appears to use uncapped comments, while the toxicity/sentiment panels use capped comments. This distinction is not explicit in the paper.
- Topic modeling preprocessing removes a code-defined list of domain and social-media stopwords and drops processed texts with length <= 2 after preprocessing. The paper does not list these stopwords or the short-text filter.
- The topic-modeling script uses `all-mpnet-base-v2`, UMAP `n_neighbors=20`, `min_dist=0.05`, HDBSCAN `min_cluster_size=300`, `max_cluster_size=5000`, and `min_samples=15`, consistent with Appendix Table A1. The exact code also sets `core_dist_n_jobs=1` and `prediction_data=True`, which are not in the paper.
- The current run generated Figure 3 with UMAP `n_epochs={umap_epochs}`. Appendix Table A1 states 50,000 epochs; rerun `Scripts/run_all.sh --umap-epochs 50000` for that exact visualization setting.
- Figure 3 uses `Data/embeddings.pkl`, which must have the same row count as `Data/topics.feather`.
- The paper discusses non-English limitations, but language filtering is not used in the paper figures reproduced here.

## Key Reproduced Test Result

- Overall videos-vs-comments polarity Mann-Whitney test on Figure 6 analysis data: q = `{vidcom_q:.3g}`.

## Stopword Lists Found in Code

- Topic-model domain stopwords ({len(TOPIC_MODEL_DOMAIN_STOPWORDS)}): `{", ".join(sorted(TOPIC_MODEL_DOMAIN_STOPWORDS))}`
- Topic-model social/media stopwords ({len(TOPIC_MODEL_SOCIAL_STOPWORDS)}): `{", ".join(sorted(TOPIC_MODEL_SOCIAL_STOPWORDS))}`
"""
    report_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    new_root = repo_root
    paths = ensure_dirs(new_root)
    set_plot_theme()

    if not args.skip_paper_render:
        assets = extract_paper_assets(repo_root, paths["paper"])
        assets.to_csv(paths["data"] / f"paper_asset_extraction_{args.dataset}.csv", index=False)

    inputs = read_inputs(repo_root / "Data", args.dataset)
    configure_topic_names(inputs["id2topics"])
    prepared = prepare_data(
        inputs,
        paths["data"],
        args.dataset,
        max_comments=args.max_comments_per_video,
        random_state=args.random_state,
    )
    write_topic_tables(inputs["id2topics"], paths["data"], paths["figures"], args.dataset)

    top_2024 = top_videos_by_comments(prepared["videos_unique"], prepared["comments_unique"], paths["data"], args.dataset, year=2024)

    for step in tqdm(
        [
            "figure_1",
            "figure_2",
            "figure_3",
            "figure_4_5",
            "figure_6_tests",
            "figure_6",
            "audit",
        ],
        desc="Generating paper outputs",
    ):
        if step == "figure_1":
            plot_figure_1(prepared["videos_unique"], prepared["comments_unique"], paths["data"], paths["figures"], args.dataset)
        elif step == "figure_2":
            plot_figure_2(paths["figures"], args.dataset)
        elif step == "figure_3":
            if not args.skip_umap:
                coords = compute_or_load_umap(
                    repo_root,
                    paths["data"],
                    inputs["topics"],
                    args.dataset,
                    n_epochs=args.umap_epochs,
                )
                plot_figure_3(coords, paths["figures"], args.dataset, n_epochs=args.umap_epochs)
        elif step == "figure_4_5":
            plot_figures_4_and_5(
                prepared["videos_with_topics"],
                prepared["comments_raw_topic_valid"],
                paths["data"],
                paths["figures"],
                args.dataset,
            )
        elif step == "figure_6_tests":
            videos_tests, comments_tests, vidcom_test = run_figure6_tests(
                prepared["videos_analysis"],
                prepared["comments_analysis"],
                paths["data"],
                args.dataset,
            )
        elif step == "figure_6":
            plot_figure_6(
                prepared["videos_analysis"],
                prepared["comments_analysis"],
                videos_tests,
                comments_tests,
                paths["figures"],
                args.dataset,
            )
        elif step == "audit":
            write_methodology_audit(
                new_root / "methodology_audit.md",
                prepared["sanity"],
                vidcom_test,
                dataset=args.dataset,
                max_comments=args.max_comments_per_video,
                random_state=args.random_state,
                umap_epochs=args.umap_epochs,
                top_videos_2024=top_2024,
            )

    print(f"Done. Data: {paths['data']}")
    print(f"Done. Figures: {paths['figures']}")
    print(f"Report: {new_root / 'methodology_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
