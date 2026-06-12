# Auto News

Auto News fetches Montreal RSS feeds, groups similar stories with Mistral, extracts article text, and generates a browser report showing comprehensive sourced coverage across providers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Add your Mistral key to `.env`:

```bash
MISTRAL_API_KEY=your_mistral_api_key_here
```

The app also supports the older `MISTRAL_KEY` name for compatibility.

## Run

Scan for new/changed news and queue background processing:

```bash
auto-news --scan
```

Run the worker to process queued scan tasks:

```bash
auto-news --worker
```

Read the cached news report in your browser:

```bash
auto-news --read
```

The default report is written to `reports/latest.html`. `--read` is cache-only: it renders existing cached topic reports and opens the browser without crawling, embedding, or calling Mistral.

Clean stale single-source reports/caches after changing grouping rules:

```bash
auto-news --cleanup
```

For step-by-step debugging, process one queued task and exit:

```bash
auto-news --worker --once
```

Or run the script directly:

```bash
python scraper.py
```

## Advanced Commands

Mostly just ingest RSS into the document DB:

```bash
auto-news --ingest-rss
```

Create RSS embeddings after ingest, in visible batches:

```bash
auto-news --embed-rss
```

Process a small batch if you are tuning rate limits:

```bash
auto-news --embed-rss --limit 25
```

Ingest RSS and then create embeddings:

```bash
auto-news --ingest-rss --embed-rss
```

Queue RSS ingestion and embedding for a background worker:

```bash
auto-news --enqueue
auto-news --worker
```

The queued workflow runs:

1. RSS ingestion
2. RSS embeddings
3. embedding-based topic grouping
4. one topic-report task per grouped story
5. cached report rendering

Render the browser report from cached topic reports only:

```bash
auto-news --render-cache --open
```

Legacy alias for reading:

```bash
auto-news --open
```

Write to a custom path:

```bash
auto-news --output reports/montreal.html
```

Save documents to a custom SQLite database:

```bash
auto-news --db data/auto_news.sqlite
```

Check an arbitrary article against stored topic embeddings:

```bash
auto-news --check-url "https://example.com/article"
```

If you know the source provider, pass it for better extraction:

```bash
auto-news --check-url "https://globalnews.ca/..." --provider global
```

The full report flow embeds RSS title/description text first, groups likely matching stories by embedding similarity, then uses Mistral only for the deeper sourced coverage report. The browser report starts with cumulative Provider Patterns across cached topics, then each topic has Cliffs Notes, provider-specific inclusions and omissions, framing differences, cautious bias signals, and a collapsible full sourced text. Hover over highlighted sentences in the full text to see which provider(s) support that sentence.

Each run stores JSON documents in SQLite:

- `rss_ingestion`: metadata for an RSS ingestion run.
- `rss_item`: feed item metadata before article extraction.
- `rss_embedding`: embedding for RSS title/description text used for topic grouping.
- `embedding_grouping`: metadata for the embedding-based topic grouping pass.
- `article`: extracted article text with provider, URL, topic, and RSS metadata.
- `article_revision`: immutable article snapshots keyed by URL and content hash.
- `article_embedding`: one-sentence English summary plus embedding for topic matching.
- `topic_embedding`: topic-level embedding built from the report title and article summaries.
- `topic_report_cache`: cached report keyed by topic and article content hashes.
- `manual_article`: arbitrary URLs checked with `--check-url`.
- `topic_report`: Mistral's sourced coverage and discrepancy report for a topic.
- `provider_bias_summary`: cumulative provider inclusion, omission, framing, and bias-signal counts across cached topic reports.
- `html_report`: generated report metadata.

Reruns are cache-first: unchanged RSS/article embeddings are reused, article text is hashed, changed articles create new `article_revision` documents with simple added/removed sentence diffs, and unchanged topic inputs reuse `topic_report_cache` instead of calling Mistral again.

Background work is stored in the `tasks` table. The worker handles RSS ingestion, RSS embedding, topic grouping, per-topic report generation, and cached report rendering. The browser-facing path renders from cache while ingestion/spider/LLM work happens separately.

Topic grouping is incremental: newly embedded RSS items are compared against existing `topic_embedding` documents as well as the current scan batch. A topic report is only generated when at least two providers have extractable article text, so single-source stories are stored but skipped for comparison.

Inspect the local document database:

```bash
sqlite3 data/auto_news.sqlite "select document_type, count(*) from documents group by document_type;"
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `MISTRAL_API_KEY` | required | Your Mistral API key. |
| `MISTRAL_MODEL` | `mistral-large-latest` | Model used for grouping and comparison. |
| `MISTRAL_EMBEDDING_MODEL` | `mistral-embed` | Model used for summary/topic embeddings. |
| `REQUEST_TIMEOUT_SECONDS` | `20` | Timeout for RSS and article requests. |
| `REPORT_PATH` | `reports/latest.html` | Default HTML report path. |
| `DOCUMENT_DB_PATH` | `data/auto_news.sqlite` | Default SQLite document database path. |
| `TOPIC_MATCH_THRESHOLD` | `0.82` | Minimum cosine similarity for arbitrary article topic matches. |
| `EMBEDDING_BATCH_SIZE` | `16` | Number of RSS items sent per embedding request. |
| `MISTRAL_MAX_RETRIES` | `5` | Retry attempts for Mistral 429/transient errors. |
| `MISTRAL_RETRY_BASE_SECONDS` | `5` | Exponential backoff base when no `Retry-After` header is present. |
| `MISTRAL_RETRY_MAX_SLEEP_SECONDS` | `120` | Maximum sleep between Mistral retries. |
| `MISTRAL_MIN_REQUEST_INTERVAL_SECONDS` | `1` | Minimum spacing between Mistral API requests. |

## Notes

- `.env` is intentionally ignored so your key does not get committed.
- Provider RSS feeds and selectors live in `scraper.py`.
- The current feed list is based on Feedspot's Montreal news RSS list.
- CTV and Montreal Gazette are not enabled because their listed RSS feeds did not return valid RSS during verification.
