import os

from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_KEY")
MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
EMBEDDING_MODEL = os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
DEFAULT_REPORT_PATH = os.getenv("REPORT_PATH", "reports/latest.html")
DEFAULT_DB_PATH = os.getenv("DOCUMENT_DB_PATH", "data/biasmeter.sqlite")
TOPIC_MATCH_THRESHOLD = float(os.getenv("TOPIC_MATCH_THRESHOLD", "0.82"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
MISTRAL_MAX_RETRIES = int(os.getenv("MISTRAL_MAX_RETRIES", "5"))
MISTRAL_MIN_REQUEST_INTERVAL_SECONDS = float(
    os.getenv("MISTRAL_MIN_REQUEST_INTERVAL_SECONDS", "1")
)

providers = {
    "citynews": {
        "selectors": ["article", ".entry-content", ".article__content", "main"],
        "rss": "https://montreal.citynews.ca/feed/",
    },
    "mtl-blog": {
        "selectors": ["article", ".article-body", ".post-content", "main"],
        "rss": "https://www.mtlblog.com/feeds/news.rss",
    },
    "global": {
        "selectors": [".l-article__story", "article", "main"],
        "rss": "https://globalnews.ca/montreal/feed/",
    },
    "suburban": {
        "selectors": [".asset-content", ".asset-body", "article", "main"],
        "rss": "https://www.thesuburban.com/search/?f=rss&t=article&c=news&l=50&s=start_time&sd=desc",
    },
    "montreal-times": {
        "selectors": [".entry-content", "article", "main"],
        "rss": "https://mtltimes.ca/feed/",
    },
    "la-presse": {
        "selectors": ["article", ".article-content", ".content", "main"],
        "rss": "https://www.lapresse.ca/actualites/grand-montreal/rss",
    },
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
