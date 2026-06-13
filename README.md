# Biasmeter

Biasmeter fetches Montreal RSS feeds, groups similar stories with Mistral, extracts article text, and generates a browser report showing comprehensive sourced coverage across providers.

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
biasmeter --scan
```

Run the worker to process queued scan tasks. It exits when no runnable tasks remain:

```bash
biasmeter --worker
```

Read the cached news report in your browser:

```bash
biasmeter --read
```

The default recent-news page is written to `reports/latest.html`, with provider breakdowns beside it at `reports/provider-breakdowns.html`. `--read` is cache-only: it renders existing cached topic reports and opens the browser without crawling, embedding, or calling Mistral.

Clean stale single-source reports/caches after changing grouping rules:

```bash
biasmeter --cleanup
```

For step-by-step debugging, process one queued task and exit:

```bash
biasmeter --worker --once
```

Keep watching for future queued work:

```bash
biasmeter --worker --watch
```

Or run the script directly:

```bash
python scraper.py
```

## Advanced Commands

Mostly just ingest RSS into the document DB:

```bash
biasmeter --ingest-rss
```

Create RSS embeddings after ingest, in visible batches:

```bash
biasmeter --embed-rss
```

Process a small batch if you are tuning rate limits:

```bash
biasmeter --embed-rss --limit 25
```

Ingest RSS and then create embeddings:

```bash
biasmeter --ingest-rss --embed-rss
```

Queue RSS ingestion and embedding for a background worker:

```bash
biasmeter --enqueue
biasmeter --worker
```

The queued workflow runs:

1. RSS ingestion
2. RSS embeddings
3. embedding-based topic grouping
4. one topic-report task per grouped story
5. cached report rendering

Render the browser report from cached topic reports only:

```bash
biasmeter --render-cache --open
```

Legacy alias for reading:

```bash
biasmeter --open
```

Write to a custom path:

```bash
biasmeter --output reports/montreal.html
```

Save documents to a custom SQLite database:

```bash
biasmeter --db data/biasmeter.sqlite
```

Check an arbitrary article against stored topic embeddings:

```bash
biasmeter --check-url "https://example.com/article"
```

If you know the source provider, pass it for better extraction:

```bash
biasmeter --check-url "https://globalnews.ca/..." --provider global
```

The full report flow embeds RSS title/description text first, groups likely matching stories by embedding similarity, then uses Mistral only for the deeper sourced coverage report. The browser GUI has two pages: Recent News for Cliffs Notes, source links, and sourced full text; Provider Breakdowns for cumulative Provider Patterns plus per-provider summaries of recurring inclusions, omissions, framing differences, cautious bias signals, and the repeated types of information each outlet tends to include or leave out. Provider summaries are refreshed whenever a new topic report is stored. Hover over highlighted sentences in the full text to see which provider(s) support that sentence.

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
- `html_report`: generated two-page browser report metadata.

Reruns are cache-first: unchanged RSS/article embeddings are reused, article text is hashed, changed articles create new `article_revision` documents with simple added/removed sentence diffs, and unchanged topic inputs reuse `topic_report_cache` instead of calling Mistral again.

Background work is stored in the `tasks` table. The worker handles RSS ingestion, RSS embedding, topic grouping, per-topic report generation, and cached report rendering, then exits when no runnable tasks remain. Use `biasmeter --worker --watch` if you want it to keep polling for future tasks. The browser-facing path renders from cache while ingestion/spider/LLM work happens separately.

Topic grouping is incremental: newly embedded RSS items are compared against existing `topic_embedding` documents as well as the current scan batch. A topic report is only generated when at least two providers have extractable article text, so single-source stories are stored but skipped for comparison.

Inspect the local document database:

```bash
sqlite3 data/biasmeter.sqlite "select document_type, count(*) from documents group by document_type;"
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `MISTRAL_API_KEY` | required | Your Mistral API key. |
| `MISTRAL_MODEL` | `mistral-large-latest` | Model used for grouping and comparison. |
| `MISTRAL_EMBEDDING_MODEL` | `mistral-embed` | Model used for summary/topic embeddings. |
| `REQUEST_TIMEOUT_SECONDS` | `20` | Timeout for RSS and article requests. |
| `REPORT_PATH` | `reports/latest.html` | Default HTML report path. |
| `DOCUMENT_DB_PATH` | `data/biasmeter.sqlite` | Default SQLite document database path. |
| `TOPIC_MATCH_THRESHOLD` | `0.82` | Minimum cosine similarity for arbitrary article topic matches. |
| `EMBEDDING_BATCH_SIZE` | `16` | Number of RSS items sent per embedding request. |
| `MAX_ARTICLES_PER_PROVIDER_PER_TOPIC` | `3` | Maximum articles from one provider in a topic comparison task. |
| `MAX_ARTICLES_PER_TOPIC` | `12` | Maximum total articles in one topic comparison task. |
| `MISTRAL_MAX_RETRIES` | `5` | Retry attempts for Mistral retryable errors that include `Retry-After`. |
| `MISTRAL_MIN_REQUEST_INTERVAL_SECONDS` | `1` | Minimum spacing between Mistral API requests. |
| `WORKER_STALE_TASK_SECONDS` | `900` | Age after which `running` tasks are reclaimed when the worker starts. |

## Notes

- `.env` is intentionally ignored so your key does not get committed.
- Provider RSS feeds and selectors live in `biasmeter/config.py`.
- The current feed list is based on Feedspot's Montreal news RSS list.
- CTV and Montreal Gazette are not enabled because their listed RSS feeds did not return valid RSS during verification.
