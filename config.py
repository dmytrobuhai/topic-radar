import os

APP_TITLE = "📡 Topic Radar - Papers • News • Videos"

DATE_WINDOWS = {
    "Last 3 days": 3,
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 365 days": 365,
}

DEFAULT_MAX_TOTAL = 30

YT_KEY = os.getenv("YT_API_KEY", "").strip()
EMAIL = os.getenv("EMAIL", "").strip()

# Minimal per-category counts
DEFAULT_MIN_PAPERS = 6
DEFAULT_MIN_VIDEOS = 3
DEFAULT_MIN_NEWS = 3

# Papers per-source mix
DEFAULT_PAPER_SPLIT = dict(arxiv=50, crossref=20, openalex=20, hf=10)

# Global scoring weights
DEFAULT_WEIGHTS = dict(
    W_TITLE_BASE=0.55,
    W_ABS_BASE=0.20,
    W_TITLE_PRF=0.15,
    W_ABS_PRF=0.05,
    W_RECENCY=0.05,
)
