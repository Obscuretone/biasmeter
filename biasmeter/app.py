import argparse
import json
import math
import re
import time
import webbrowser
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError

from biasmeter.config import (
    DEFAULT_DB_PATH,
    DEFAULT_REPORT_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    MAX_ARTICLES_PER_PROVIDER_PER_TOPIC,
    MAX_ARTICLES_PER_TOPIC,
    MISTRAL_API_KEY,
    MISTRAL_MAX_RETRIES,
    MISTRAL_MIN_REQUEST_INTERVAL_SECONDS,
    MODEL,
    REQUEST_TIMEOUT_SECONDS,
    TOPIC_MATCH_THRESHOLD,
    WORKER_STALE_TASK_SECONDS,
    headers,
    providers,
)
from biasmeter.report import (
    build_provider_patterns,
    render_cached_report,
    render_html_report,
)
from biasmeter.store import (
    DocumentStore,
    get_cached_topic_report,
    stable_json_hash,
    store_article_revision,
    store_topic_report_cache,
)

UTC = timezone.utc
last_mistral_request_at = 0.0


def progress(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


class MistralRetryAfterError(RuntimeError):
    def __init__(self, operation_name, retry_after_seconds, original_error):
        self.operation_name = operation_name
        self.retry_after_seconds = retry_after_seconds
        self.original_error = original_error
        super().__init__(
            f"{operation_name} rate-limited by Mistral; retry after "
            f"{retry_after_seconds:.3f}s. Original error: {original_error}"
        )


class MistralRateLimitWithoutRetryAfterError(RuntimeError):
    def __init__(self, operation_name, original_error):
        self.operation_name = operation_name
        self.original_error = original_error
        super().__init__(
            f"{operation_name} rate-limited by Mistral without Retry-After. "
            f"Original error: {original_error}"
        )


def get_mistral_client():
    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "Missing Mistral API key. Set MISTRAL_API_KEY in your environment or .env file."
        )

    return Mistral(api_key=MISTRAL_API_KEY)


def extract_mistral_text(chat_response: Any) -> str:
    if chat_response is None:
        raise RuntimeError("Mistral response was empty")

    content = chat_response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Mistral response content was empty")

    return content.strip()


def throttle_mistral_request():
    global last_mistral_request_at

    if MISTRAL_MIN_REQUEST_INTERVAL_SECONDS <= 0:
        return

    elapsed = time.monotonic() - last_mistral_request_at
    wait_seconds = MISTRAL_MIN_REQUEST_INTERVAL_SECONDS - elapsed

    if wait_seconds > 0:
        time.sleep(wait_seconds)

    last_mistral_request_at = time.monotonic()


def get_exception_response(exc):
    raw_response = getattr(exc, "raw_response", None)

    if raw_response is None and isinstance(exc, SDKError):
        raw_response = exc.raw_response

    return raw_response


def get_header_value(headers, header_name):
    if not headers:
        return None

    try:
        value = headers.get(header_name)
        if value is not None:
            return value
    except AttributeError:
        pass

    lowered_header_name = header_name.lower()
    try:
        header_items = headers.items()
    except AttributeError:
        return None

    for key, value in header_items:
        if str(key).lower() == lowered_header_name:
            return value

    return None


def parse_retry_after_seconds(retry_after):
    if not retry_after:
        return None

    value = str(retry_after).strip()
    try:
        return max(float(value), 0.001)
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    else:
        retry_at = retry_at.astimezone(UTC)

    delay_seconds = (retry_at - datetime.now(UTC)).total_seconds()
    if delay_seconds <= 0:
        return None

    return max(delay_seconds, 0.001)


def parse_retry_after_from_message(message):
    lowered_message = str(message).lower()
    retry_patterns = (
        r"retry(?:-|\s*)after[:=\s]+(\d+(?:\.\d+)?)\s*(ms|millisecond|milliseconds|s|sec|secs|second|seconds|m|min|mins|minute|minutes)?",
        r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|millisecond|milliseconds|s|sec|secs|second|seconds|m|min|mins|minute|minutes)?",
        r"wait\s+(\d+(?:\.\d+)?)\s*(ms|millisecond|milliseconds|s|sec|secs|second|seconds|m|min|mins|minute|minutes)",
    )

    for pattern in retry_patterns:
        match = re.search(pattern, lowered_message)
        if not match:
            continue

        delay = float(match.group(1))
        unit = match.group(2) or "seconds"
        if unit in {"ms", "millisecond", "milliseconds"}:
            delay = delay / 1000
        elif unit in {"m", "min", "mins", "minute", "minutes"}:
            delay = delay * 60

        return max(delay, 0.001)

    return None


def get_retry_after_seconds(exc):
    raw_response = get_exception_response(exc)

    if raw_response is not None:
        headers = getattr(raw_response, "headers", None)
        retry_after = get_header_value(headers, "retry-after")
        retry_after_seconds = parse_retry_after_seconds(retry_after)

        if retry_after_seconds is not None:
            return retry_after_seconds

    return parse_retry_after_from_message(exc)


def is_mistral_rate_limit_error(exc):
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    raw_response = get_exception_response(exc)
    if getattr(raw_response, "status_code", None) == 429:
        return True

    message = str(exc).lower()
    return "rate limit" in message or "too many requests" in message


def is_mistral_transient_error(exc):
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 425, 500, 502, 503, 504}:
        return True

    raw_response = get_exception_response(exc)
    if getattr(raw_response, "status_code", None) in {408, 425, 500, 502, 503, 504}:
        return True

    message = str(exc).lower()
    transient_markers = (
        "connection refused",
        "connection reset",
        "connection aborted",
        "timed out",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )
    return any(marker in message for marker in transient_markers)


def call_mistral_with_retries(operation_name, callback):
    for retry_count in range(1, MISTRAL_MAX_RETRIES + 2):
        try:
            throttle_mistral_request()
            return callback()
        except Exception as exc:
            rate_limited = is_mistral_rate_limit_error(exc)
            retryable = rate_limited or is_mistral_transient_error(exc)

            if not retryable:
                raise

            delay_seconds = get_retry_after_seconds(exc)
            if delay_seconds is None:
                print(
                    f"{operation_name} hit a retryable Mistral error "
                    f"({exc}) but did not include Retry-After. Not retrying."
                )
                if rate_limited:
                    raise MistralRateLimitWithoutRetryAfterError(
                        operation_name, exc
                    ) from exc
                raise

            if retry_count > MISTRAL_MAX_RETRIES:
                raise MistralRetryAfterError(
                    operation_name, delay_seconds, exc
                ) from exc

            print(
                f"{operation_name} hit a retryable Mistral error "
                f"({exc}). Waiting {delay_seconds:.3f}s from Retry-After "
                f"before retry {retry_count}/{MISTRAL_MAX_RETRIES}."
            )
            time.sleep(delay_seconds)


# Function to fetch and parse an RSS feed
def fetch_rss(url, provider):
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)

    rss_items = []  # List to hold all the parsed RSS feed items

    if response.status_code == 200:
        # Parse the XML content
        root = ElementTree.fromstring(response.content)

        # Extract the data and store it in the list
        for item in root.findall(".//item"):
            title = item.findtext("title", default="").strip()
            link = item.findtext("link", default="").strip()
            description = item.findtext("description", default="").strip()
            pub_date = item.findtext("pubDate", default="").strip()

            if not title or not link:
                continue

            feed_item = {
                "title": title,
                "link": link,
                "description": BeautifulSoup(description, "html.parser").get_text(
                    " ", strip=True
                ),
                "pub_date": pub_date,
                "provider": provider,
            }
            rss_items.append(feed_item)
    else:
        print(
            f"Failed to retrieve RSS feed from {url}. Status code: {response.status_code}"
        )

    return rss_items


# Function to fetch all RSS data from all feeds
def get_all_rss_data():
    all_rss_data = []  # List to store data from all feeds

    for provider in providers:
        rss_url = providers[provider].get("rss")
        print(f"Fetching feed from: {rss_url}")
        rss_data = fetch_rss(rss_url, provider)

        all_rss_data.extend(rss_data)  # Add items from this feed to the main list

    return all_rss_data


def ingest_rss(store):
    rss_data = get_all_rss_data()
    counts_by_provider = {}

    for item in rss_data:
        store.upsert("rss_item", item["link"], item)
        provider = item["provider"]
        counts_by_provider[provider] = counts_by_provider.get(provider, 0) + 1

    store.upsert(
        "rss_ingestion",
        datetime.now().isoformat(timespec="seconds"),
        {
            "item_count": len(rss_data),
            "counts_by_provider": counts_by_provider,
            "providers": sorted(counts_by_provider.keys()),
        },
    )

    print(f"\nSaved {len(rss_data)} RSS items.")
    for provider, count in sorted(counts_by_provider.items()):
        print(f"- {provider}: {count}")

    return rss_data


def extract_content_by_selector(html, selectors):
    """
    Extracts the content of the article using a CSS selector with BeautifulSoup.
    """
    soup = BeautifulSoup(html, "html.parser")

    for selector in selectors:
        print(f"Extracting content using {selector}")
        content = soup.select_one(selector)

        if content:
            response = content.get_text(" ", strip=True)
            if response:
                return response

    return None


def extract_article_from_url(url, provider=None):
    provider_config = providers.get(provider) if provider else None
    selectors = (
        provider_config.get("selectors", [])
        if provider_config
        else [
            "article",
            "main",
            ".article-body",
            ".entry-content",
            ".post-content",
            ".content",
        ]
    )
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    content = extract_content_by_selector(response.text, selectors)

    if not content:
        raise RuntimeError(f"Could not extract article content from {url}")

    soup = BeautifulSoup(response.text, "html.parser")
    title = ""

    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    return {
        "provider": provider or "manual",
        "url": url,
        "title": title,
        "description": "",
        "pub_date": "",
        "topic": "",
        "content": content,
    }


# Function to format the articles as a prompt to send to Mistral
def format_articles_for_llm(articles):
    formatted_articles = []
    for article in articles:
        formatted_articles.append(
            f"Title: {article['title']}\n"
            f"Description: {article['description']}\n"
            f"Provider: {article['provider']}\n"
            f"URL: {article['link']}"
        )

    return "\n\n".join(formatted_articles)


# Function to send the formatted data to Mistral to analyze similarity
def send_to_llm_for_grouping(articles):
    print("going to LLM to ask what articles are about the same thing")
    # Format the articles as a prompt
    prompt = (
        "Here are several articles. Tell me which articles are about the same thing "
        "and give me only their URLs as structured JSON. The output should include "
        "the news provider and the link. "
        "Give me an array by topic using the topic as the key, and in each one, "
        "the keys should be 'provider' and 'url'. "
        f"provider should be one of: {', '.join(providers.keys())}. "
        ""
        ":\n\n"
        f"{format_articles_for_llm(articles)}"
    )
    # Send the request to Mistral
    try:
        client = get_mistral_client()
        chat_response = call_mistral_with_retries(
            "Mistral grouping request",
            lambda: client.chat.complete(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={
                    "type": "json_object",
                },
            ),
        )
        # Return the LLM's response (the similarity analysis)
        return extract_mistral_text(chat_response)
    except (MistralRetryAfterError, MistralRateLimitWithoutRetryAfterError):
        raise
    except Exception as e:
        print(f"Error while sending to Mistral: {e}")
        return None


def parse_llm_json(content):
    content = content.strip()

    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").strip()
        content = content.removesuffix("```").strip()

    return json.loads(content)


def summarize_article_for_matching(article):
    """
    Create a compact factual summary used for topic matching.
    """
    print(f"Summarizing article for matching: {article['url']}")
    try:
        client = get_mistral_client()
        chat_response = call_mistral_with_retries(
            "Mistral article summary request",
            lambda: client.chat.complete(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write compact factual summaries for topic matching. "
                            "Write exactly one English sentence. "
                            "Include the main event, location, key people or organizations, and outcome if present. "
                            "Do not add facts that are not in the article."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Title: {article.get('title', '')}\n"
                            f"Provider: {article.get('provider', '')}\n"
                            f"Article text:\n{article.get('content', '')[:6000]}"
                        ),
                    },
                ],
            ),
        )
        return extract_mistral_text(chat_response)
    except (MistralRetryAfterError, MistralRateLimitWithoutRetryAfterError):
        raise
    except Exception as e:
        print(f"Error summarizing article: {e}")
        return article.get("description") or article.get("title") or ""


def create_embedding(text):
    if not text:
        return []

    return create_embeddings([text])[0]


def create_embeddings(texts):
    if not texts:
        return []

    progress(f"Requesting embeddings for {len(texts)} item(s).")
    client = get_mistral_client()
    response = call_mistral_with_retries(
        "Mistral embedding request",
        lambda: client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=texts,
        ),
    )
    if response is None:
        raise RuntimeError("Mistral embedding response was empty")
    progress(f"Received embeddings for {len(response.data)} item(s).")
    return [item.embedding for item in response.data]


def batched(items, batch_size):
    batch_size = max(batch_size, 1)
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def rss_item_embedding_text(item):
    return " ".join(
        part
        for part in [
            item.get("title", ""),
            item.get("description", ""),
            item.get("provider", ""),
        ]
        if part
    ).strip()


def enrich_rss_items_with_embeddings(store, rss_data):
    enriched_items = []
    uncached_items = []
    uncached_texts = []

    for item in rss_data:
        cached_embedding = store.get("rss_embedding", item["link"])

        if cached_embedding:
            enriched_item = {
                **item,
                "embedding_text": cached_embedding.get("embedding_text", ""),
                "embedding": cached_embedding.get("embedding", []),
            }
            enriched_items.append(enriched_item)
            continue

        embedding_text = rss_item_embedding_text(item)
        if not embedding_text:
            continue

        uncached_items.append(item)
        uncached_texts.append(embedding_text)

    if uncached_texts:
        print(f"Embedding {len(uncached_texts)} RSS items for topic grouping")
        embeddings = []
        text_batches = list(batched(uncached_texts, EMBEDDING_BATCH_SIZE))

        for batch_index, text_batch in enumerate(text_batches, start=1):
            print(
                f"Embedding RSS batch {batch_index}/{len(text_batches)} "
                f"({len(text_batch)} items)"
            )
            embeddings.extend(create_embeddings(text_batch))

        for item, embedding_text, embedding in zip(
            uncached_items, uncached_texts, embeddings, strict=False
        ):
            enriched_item = {
                **item,
                "embedding_text": embedding_text,
                "embedding": embedding,
            }
            store.upsert(
                "rss_embedding",
                item["link"],
                {
                    "provider": item.get("provider"),
                    "url": item.get("link"),
                    "title": item.get("title"),
                    "embedding_text": embedding_text,
                    "embedding_model": EMBEDDING_MODEL,
                    "embedding": embedding,
                },
            )
            enriched_items.append(enriched_item)

    return enriched_items


def get_stored_rss_items(store, limit=None):
    rows = store.list_by_type("rss_item")
    items = [row["content"] for row in rows]

    if limit is not None:
        return items[:limit]

    return items


def embed_stored_rss_items(store, limit=None):
    rss_items = get_stored_rss_items(store, limit)

    if not rss_items:
        print("No RSS items found. Run biasmeter --ingest-rss first.")
        return []

    before_count = len(store.list_by_type("rss_embedding"))
    enriched_items = enrich_rss_items_with_embeddings(store, rss_items)
    after_count = len(store.list_by_type("rss_embedding"))
    created_count = max(after_count - before_count, 0)

    print(
        f"\nRSS embeddings ready: {after_count}. "
        f"Created {created_count} new embedding document(s)."
    )
    return enriched_items


def mean_embedding(vectors):
    if not vectors:
        return []

    vector_length = len(vectors[0])
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(vector_length)
    ]


def article_reference(provider, url, similarity_basis=""):
    return {
        "provider": provider,
        "url": url,
        "similarity_basis": similarity_basis,
    }


def dedupe_article_references(articles):
    deduped = {}
    for article in articles:
        provider = article.get("provider")
        url = article.get("url")
        if not provider or not url:
            continue

        deduped[(provider, url)] = article_reference(
            provider,
            url,
            article.get("similarity_basis", ""),
        )

    return list(deduped.values())


def source_articles_from_report(report):
    articles = []
    for entry in report.get("full_text", []):
        for source in entry.get("sources", []):
            provider = source.get("provider")
            url = source.get("url")
            if provider and url:
                articles.append(article_reference(provider, url, "cached topic report"))

    return articles


def source_providers_for_report(report):
    providers_in_report = set()
    for entry in report.get("full_text", []):
        for source in entry.get("sources", []):
            provider = source.get("provider")
            if provider:
                providers_in_report.add(provider)

    return providers_in_report


def get_existing_topic_articles(store, topic):
    articles = []

    for row in store.list_by_type("article"):
        article = row["content"]
        if article.get("topic") == topic:
            articles.append(
                article_reference(
                    article.get("provider"),
                    article.get("url"),
                    "stored article",
                )
            )

    report = store.get("topic_report", topic)
    if report:
        articles.extend(source_articles_from_report(report))

    return dedupe_article_references(articles)


def providers_for_articles(articles):
    return {article.get("provider") for article in articles if article.get("provider")}


def limit_topic_articles(
    articles,
    max_per_provider=MAX_ARTICLES_PER_PROVIDER_PER_TOPIC,
    max_total=MAX_ARTICLES_PER_TOPIC,
):
    kept_articles = []
    counts_by_provider = {}

    for article in articles:
        provider = article.get("provider")
        if not provider:
            continue

        provider_count = counts_by_provider.get(provider, 0)
        if provider_count >= max_per_provider:
            continue

        kept_articles.append(article)
        counts_by_provider[provider] = provider_count + 1

        if len(kept_articles) >= max_total:
            break

    return kept_articles


def cleanup_single_source_artifacts(store):
    removed_reports = 0
    removed_report_caches = 0
    removed_topic_embeddings = 0
    removed_report_tasks = 0

    for row in store.list_by_type("topic_report"):
        report = row["content"]
        source_providers = source_providers_for_report(report)
        if len(source_providers) >= 2:
            continue

        topic_key = row["document_key"]
        title = report.get("title")
        topic = report.get("topic")
        store.delete_document("topic_report", topic_key)
        removed_reports += 1

        for embedding_key in {topic_key, title, topic}:
            if embedding_key:
                store.delete_document("topic_embedding", embedding_key)
                removed_topic_embeddings += 1

    for row in store.list_by_type("topic_report_cache"):
        report = row["content"].get("report", {})
        source_providers = source_providers_for_report(report)
        if len(source_providers) < 2:
            store.delete_document("topic_report_cache", row["document_key"])
            removed_report_caches += 1

    for task in store.list_tasks("generate_topic_report"):
        source_providers = providers_for_articles(task["payload"].get("articles", []))
        if len(source_providers) < 2:
            store.delete_task(task["id"])
            removed_report_tasks += 1

    print("Cleanup complete.")
    print(
        "Removed "
        f"{removed_reports} topic report(s), "
        f"{removed_report_caches} report cache(s), "
        f"{removed_topic_embeddings} topic embedding key(s), and "
        f"{removed_report_tasks} queued single-source report task(s)."
    )
    return {
        "topic_reports": removed_reports,
        "topic_report_caches": removed_report_caches,
        "topic_embedding_keys": removed_topic_embeddings,
        "report_tasks": removed_report_tasks,
    }


def group_rss_items_by_embedding(store, rss_data, threshold=TOPIC_MATCH_THRESHOLD):
    enriched_items = enrich_rss_items_with_embeddings(store, rss_data)
    clusters = []
    existing_topic_groups = {}

    for item in enriched_items:
        existing_topic_matches = find_similar_topic_embeddings(
            store, item["embedding"], threshold
        )

        if existing_topic_matches:
            topic = existing_topic_matches[0].get("topic") or existing_topic_matches[
                0
            ].get("title")
            if topic:
                existing_topic_groups.setdefault(
                    topic,
                    get_existing_topic_articles(store, topic),
                )
                existing_topic_groups[topic].append(
                    article_reference(
                        item["provider"],
                        item["link"],
                        item["embedding_text"],
                    )
                )
                continue

        best_cluster = None
        best_similarity = 0.0

        for cluster in clusters:
            centroid_similarity = cosine_similarity(
                item["embedding"], cluster["centroid"]
            )
            seed_similarity = cosine_similarity(
                item["embedding"], cluster["seed_embedding"]
            )
            if (
                centroid_similarity >= threshold
                and seed_similarity >= threshold
                and centroid_similarity > best_similarity
            ):
                best_cluster = cluster
                best_similarity = centroid_similarity

        if best_cluster:
            best_cluster["items"].append(item)
            best_cluster["centroid"] = mean_embedding(
                [cluster_item["embedding"] for cluster_item in best_cluster["items"]]
            )
        else:
            clusters.append(
                {
                    "topic": item["title"],
                    "seed_embedding": item["embedding"],
                    "centroid": item["embedding"],
                    "items": [item],
                }
            )

    grouped_articles = {}
    for topic, articles in existing_topic_groups.items():
        articles = dedupe_article_references(articles)
        if len(providers_for_articles(articles)) < 2:
            continue

        limited_articles = limit_topic_articles(articles)
        if len(providers_for_articles(limited_articles)) < 2:
            continue

        grouped_articles[topic] = limited_articles

    for cluster in clusters:
        providers_in_cluster = {item["provider"] for item in cluster["items"]}
        if len(providers_in_cluster) < 2:
            continue

        topic = cluster["topic"]
        sorted_items = sorted(
            cluster["items"],
            key=lambda cluster_item: cosine_similarity(
                cluster_item["embedding"], cluster["seed_embedding"]
            ),
            reverse=True,
        )
        limited_articles = limit_topic_articles(
            [
                article_reference(
                    item["provider"], item["link"], item["embedding_text"]
                )
                for item in sorted_items
            ]
        )
        if len(providers_for_articles(limited_articles)) < 2:
            continue

        grouped_articles[topic] = limited_articles

    store.upsert(
        "embedding_grouping",
        datetime.now().isoformat(timespec="seconds"),
        {
            "threshold": threshold,
            "cluster_count": len(clusters),
            "existing_topic_group_count": len(existing_topic_groups),
            "multi_provider_group_count": len(grouped_articles),
            "topics": list(grouped_articles.keys()),
        },
    )
    print(
        f"Embedding grouping found {len(grouped_articles)} multi-provider topic groups."
    )
    return grouped_articles


def cosine_similarity(first_vector, second_vector):
    if not first_vector or not second_vector:
        return 0.0

    dot_product = sum(
        first_value * second_value
        for first_value, second_value in zip(first_vector, second_vector, strict=False)
    )
    first_norm = math.sqrt(sum(value * value for value in first_vector))
    second_norm = math.sqrt(sum(value * value for value in second_vector))

    if not first_norm or not second_norm:
        return 0.0

    return dot_product / (first_norm * second_norm)


def find_similar_topic_embeddings(store, embedding, threshold=TOPIC_MATCH_THRESHOLD):
    matches = []
    for row in store.list_by_type("topic_embedding"):
        topic_document = row["content"]
        similarity = cosine_similarity(embedding, topic_document.get("embedding", []))

        if similarity >= threshold:
            matches.append(
                {
                    "topic": topic_document.get("topic"),
                    "title": topic_document.get("title"),
                    "similarity": similarity,
                }
            )

    return sorted(matches, key=lambda item: item["similarity"], reverse=True)


def enrich_article_with_summary_embedding(store, article):
    cached_embedding = store.get("article_embedding", article["url"])

    if cached_embedding:
        article["summary"] = cached_embedding.get("summary", "")
        article["embedding"] = cached_embedding.get("embedding", [])
        article["topic_matches"] = find_similar_topic_embeddings(
            store, article["embedding"]
        )
        return article

    summary = summarize_article_for_matching(article)
    embedding = create_embedding(summary)
    topic_matches = find_similar_topic_embeddings(store, embedding)

    article["summary"] = summary
    article["embedding"] = embedding
    article["topic_matches"] = topic_matches
    store.upsert(
        "article_embedding",
        article["url"],
        {
            "provider": article.get("provider"),
            "url": article.get("url"),
            "title": article.get("title"),
            "topic": article.get("topic"),
            "summary": summary,
            "embedding_model": EMBEDDING_MODEL,
            "embedding": embedding,
            "topic_matches": topic_matches,
        },
    )
    return article


def build_topic_embedding(store, report, articles_content):
    topic_text = " ".join(
        [
            report.get("title", ""),
            report.get("topic", ""),
            " ".join(article.get("summary", "") for article in articles_content),
        ]
    ).strip()

    if not topic_text:
        return None

    embedding = create_embedding(topic_text)
    document = {
        "topic": report.get("topic"),
        "title": report.get("title"),
        "summary": topic_text,
        "embedding_model": EMBEDDING_MODEL,
        "embedding": embedding,
    }
    store.upsert(
        "topic_embedding", report.get("topic") or report.get("title"), document
    )
    return document


def refresh_provider_bias_summary(store):
    reports = [row["content"] for row in store.list_by_type("topic_report")]
    provider_patterns = build_provider_patterns(reports)
    store.upsert(
        "provider_bias_summary",
        "latest",
        {
            "generated_at": datetime.now().isoformat(),
            "topic_count": len(reports),
            "providers": provider_patterns,
        },
    )
    return provider_patterns


def send_to_llm_for_comparison(topic, articles_content):
    """
    Send the articles' content to Mistral to flag discrepancies across providers.
    """
    providers_in_content = sorted(providers_for_articles(articles_content))
    progress(
        "Finding discrepancies for "
        f"{topic} using {len(articles_content)} article(s) "
        f"from {', '.join(providers_in_content)}."
    )
    try:
        client = get_mistral_client()
        formatted_content = "\n\n".join(
            [
                f"Provider: {article['provider']}\n"
                f"URL: {article['url']}\n"
                f"Content: {article['content']}"
                for article in articles_content
            ]
        )

        chat_response = call_mistral_with_retries(
            "Mistral coverage report request",
            lambda: client.chat.complete(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a media coverage analyst comparing news articles about the same topic. "
                            "Your goal is to create comprehensive English coverage of every material detail in the source articles, while identifying what each provider included or omitted. "
                            "Do not write a conventional rewritten article and do not smooth over differences between providers. "
                            "Do not hallucinate or infer facts not present in the provided article text. "
                            "Treat omissions and inclusions as signals, not proof of intent. "
                            "Every sentence in the full_text array must be grounded in one or more listed sources. "
                            "Always write in English, even when source articles are in French."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Topic: {topic}\n\n"
                            "Return only valid JSON with this exact shape:\n"
                            "{\n"
                            '  "title": "short English topic title",\n'
                            '  "cliffs_notes": [\n'
                            '    "short English bullet describing what happened, with no source-comparison commentary"\n'
                            "  ],\n"
                            '  "full_text": [\n'
                            "    {\n"
                            '      "sentence": "one English sentence containing a material fact from the coverage",\n'
                            '      "sources": [\n'
                            '        {"provider": "provider name", "url": "source URL", "support": "brief supporting detail from that provider"}\n'
                            "      ]\n"
                            "    }\n"
                            "  ],\n"
                            '  "provider_specific_inclusions": [\n'
                            '    {"provider": "provider name", "information_type": "short label for kind of information included", "detail": "detail this provider included", "absent_from": ["provider name"], "why_it_matters": "why this may affect reader understanding"}\n'
                            "  ],\n"
                            '  "provider_specific_omissions": [\n'
                            '    {"provider": "provider name", "information_type": "short label for kind of information omitted", "missing_detail": "detail omitted or not present", "covered_by": ["provider name"], "why_it_matters": "why this may affect reader understanding"}\n'
                            "  ],\n"
                            '  "framing_differences": [\n'
                            '    {"providers": ["provider name"], "framing_axis": "short label for kind of framing difference", "difference": "difference in emphasis, wording, attribution, ordering, quoted voices, or tone"}\n'
                            "  ],\n"
                            '  "potential_bias_signals": [\n'
                            '    {"signal": "cautious evidence-backed observation", "bias_axis": "short label for suspected recurring bias type", "providers": ["provider name"], "evidence": "what in the supplied text supports this", "confidence": "high|medium|low"}\n'
                            "  ],\n"
                            '  "confidence": "high|medium|low"\n'
                            "}\n\n"
                            "Make cliffs_notes the main reader-facing summary: 4 to 7 concise factual bullets about what happened. "
                            "Do not mention providers, coverage differences, omissions, inclusions, framing, or bias in cliffs_notes. "
                            "Save all source-comparison analysis for the provider-specific and bias sections. "
                            "Use stable information_type, framing_axis, and bias_axis labels so long-term patterns can be compared across topics. "
                            "Make full_text comprehensive: include all material facts that appear in any source, especially details other providers left out. "
                            "Do not include any full_text sentence without at least one source object. "
                            "Do not claim motive or intent. If a bias signal is weak, label confidence low.\n\n"
                            f"{formatted_content}"
                        ),
                    },
                ],
                response_format={
                    "type": "json_object",
                },
            ),
        )
        progress(f"Received discrepancy report for {topic}.")
        return parse_llm_json(extract_mistral_text(chat_response))
    except (MistralRetryAfterError, MistralRateLimitWithoutRetryAfterError):
        raise
    except Exception as e:
        print(f"Error sending for comparison: {e}")
        return None


def generate_topic_report(store, topic, articles, output_path=None):
    articles_content = []
    progress(f"Starting topic report: {topic} ({len(articles)} candidate article(s)).")

    for index, article in enumerate(articles, start=1):
        provider = article.get("provider")
        url = article.get("url")
        provider_config = providers.get(provider)
        selectors = provider_config.get("selectors", []) if provider_config else []
        rss_item = store.get("rss_item", url) or {}

        if not provider or not url or not selectors:
            progress(
                f"Skipping article {index}/{len(articles)} for {topic}: missing provider, URL, or selectors."
            )
            continue

        try:
            progress(
                f"Extracting article {index}/{len(articles)} for {topic}: {provider} {url}"
            )
            article_content = extract_content_by_selector(
                requests.get(
                    url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
                ).text,
                selectors,
            )

            if article_content:
                article_document = {
                    "provider": provider,
                    "url": url,
                    "title": rss_item.get("title", ""),
                    "description": rss_item.get("description", ""),
                    "pub_date": rss_item.get("pub_date", ""),
                    "topic": topic,
                    "content": article_content,
                }
                article_document = enrich_article_with_summary_embedding(
                    store, article_document
                )
                article_document = store_article_revision(store, article_document)
                articles_content.append(article_document)
                progress(
                    f"Prepared article {index}/{len(articles)} for {topic}: {provider}"
                )
            else:
                progress(
                    f"No extractable text for article {index}/{len(articles)} for {topic}: {url}"
                )
        except (MistralRetryAfterError, MistralRateLimitWithoutRetryAfterError):
            raise
        except Exception as e:
            print(f"Error extracting content for article {url}: {e}")

    if not articles_content:
        print(f"No extractable articles for topic: {topic}")
        return None

    extracted_providers = providers_for_articles(articles_content)
    progress(
        f"Extracted {len(articles_content)} article(s) for {topic} "
        f"from {len(extracted_providers)} provider(s)."
    )
    if len(extracted_providers) < 2:
        print(
            f"Skipping topic with only one extracted provider: {topic} "
            f"({', '.join(sorted(extracted_providers)) or 'none'})"
        )
        return None

    report = get_cached_topic_report(store, topic, articles_content)

    if not report:
        report = send_to_llm_for_comparison(topic, articles_content)
    else:
        progress(f"Using cached discrepancy report for {topic}.")

    if not report:
        return None

    report["topic"] = topic
    report = store_topic_report_cache(store, topic, articles_content, report)
    store.upsert("topic_report", topic, report)
    build_topic_embedding(store, report, articles_content)
    refresh_provider_bias_summary(store)
    if output_path:
        progress(f"Rendering cached pages after storing topic report: {topic}.")
        render_cached_report(store, output_path)
    progress(f"Stored topic report: {report.get('title') or topic}")
    return report


def enqueue_topic_report_tasks(store, output_path=DEFAULT_REPORT_PATH):
    rss_items = get_stored_rss_items(store)

    if not rss_items:
        print("No RSS items found. Queue or run RSS ingestion first.")
        return {}

    grouped_articles = group_rss_items_by_embedding(store, rss_items)

    for topic, articles in grouped_articles.items():
        task_key = f"{topic}#{stable_json_hash(articles)[:12]}"
        store.enqueue_task(
            "generate_topic_report",
            task_key,
            {
                "topic": topic,
                "articles": articles,
                "output": output_path,
            },
            max_attempts=3,
        )

    store.enqueue_task(
        "render_cached_report",
        "latest",
        {
            "output": output_path,
        },
    )
    print(f"Queued {len(grouped_articles)} topic report task(s).")
    print(f"Task counts: {store.task_counts()}")
    return grouped_articles


def enqueue_background_work(store):
    store.enqueue_task("ingest_rss", "default", {})
    store.enqueue_task("embed_rss", "default", {})
    store.enqueue_task("group_topics", "default", {})
    print("Queued RSS ingestion, RSS embedding, and topic grouping tasks.")
    print(f"Task counts: {store.task_counts()}")


def process_task(store, task):
    task_type = task["task_type"]
    payload = task["payload"]

    progress(f"Running task {task['id']}: {task_type}:{task['task_key']}")

    if task_type == "ingest_rss":
        ingest_rss(store)
        return

    if task_type == "embed_rss":
        embed_stored_rss_items(store, payload.get("limit"))
        return

    if task_type == "group_topics":
        enqueue_topic_report_tasks(store, payload.get("output", DEFAULT_REPORT_PATH))
        return

    if task_type == "generate_topic_report":
        generate_topic_report(
            store,
            payload["topic"],
            payload["articles"],
            payload.get("output"),
        )
        return

    if task_type == "render_cached_report":
        progress("Rendering cached report pages.")
        render_cached_report(store, payload.get("output", DEFAULT_REPORT_PATH))
        return

    raise RuntimeError(f"Unknown task type: {task_type}")


def run_worker(store, once=False, sleep_seconds=5, stale_task_seconds=900):
    print("Worker started. Press Ctrl+C to stop.")
    recovered_count = store.requeue_stale_running_tasks(stale_task_seconds)
    if recovered_count:
        print(f"Recovered {recovered_count} stale running task(s).")

    while True:
        task = store.claim_task()

        if not task:
            if once:
                print("No pending tasks.")
                return

            time.sleep(sleep_seconds)
            continue

        try:
            process_task(store, task)
        except MistralRetryAfterError as exc:
            print(f"Task paused: {exc}")
            store.fail_task(task, exc, retry_after_seconds=exc.retry_after_seconds)
        except MistralRateLimitWithoutRetryAfterError as exc:
            print(f"Task failed without retry: {exc}")
            task["attempts"] = task["max_attempts"]
            store.fail_task(task, exc)
        except Exception as exc:
            print(f"Task failed: {exc}")
            store.fail_task(task, exc)
        else:
            store.complete_task(task["id"])
            progress(f"Completed task {task['id']}: {task['task_type']}.")

        if once:
            return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch Montreal news feeds and generate a sourced coverage report."
    )
    parser.add_argument(
        "--ingest-rss",
        action="store_true",
        help="Only fetch RSS feeds and save feed items to the document database.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Queue background work to scan feeds and add new/changed news.",
    )
    parser.add_argument(
        "--read",
        action="store_true",
        help="Open the cached browser report without crawling or calling Mistral.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove stale single-source reports/caches and queued single-source report tasks.",
    )
    parser.add_argument(
        "--enqueue",
        action="store_true",
        help="Queue background RSS ingestion and embedding tasks, then exit.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run the background task worker.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="With --worker, process one task and exit.",
    )
    parser.add_argument(
        "--stale-task-seconds",
        type=int,
        default=WORKER_STALE_TASK_SECONDS,
        help=(
            "With --worker, reclaim running tasks older than this many seconds. "
            f"Defaults to {WORKER_STALE_TASK_SECONDS}."
        ),
    )
    parser.add_argument(
        "--render-cache",
        action="store_true",
        help="Render the browser report from cached topic reports only.",
    )
    parser.add_argument(
        "--embed-rss",
        action="store_true",
        help="Create missing embeddings for stored RSS items, then exit unless running a full report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit items processed by commands like --embed-rss.",
    )
    parser.add_argument(
        "--check-url",
        help="Extract one article URL, embed its one-sentence summary, and compare it to stored topics.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(providers.keys()),
        help="Optional known provider for --check-url extraction selectors.",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"SQLite document database path. Defaults to {DEFAULT_DB_PATH}.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_REPORT_PATH,
        help=f"HTML report path. Defaults to {DEFAULT_REPORT_PATH}.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated report in your default browser.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="biasmeter 0.1.0",
    )
    return parser.parse_args()


def check_url_against_topics(store, url, provider=None):
    article = extract_article_from_url(url, provider)
    article = enrich_article_with_summary_embedding(store, article)
    store.upsert("manual_article", url, article)

    print("\nSummary:")
    print(article.get("summary", ""))

    matches = article.get("topic_matches", [])
    if not matches:
        print(f"\nNo stored topic matched above {TOPIC_MATCH_THRESHOLD:.2f}.")
        return

    print(f"\nLikely topic matches above {TOPIC_MATCH_THRESHOLD:.2f}:")
    for match in matches[:10]:
        title = match.get("title") or match.get("topic")
        print(f"- {match['similarity']:.3f}: {title}")


# Main execution
def main():
    args = parse_args()
    store = DocumentStore(args.db)

    if args.cleanup:
        try:
            cleanup_single_source_artifacts(store)
        finally:
            store.close()
        return

    if args.check_url:
        try:
            check_url_against_topics(store, args.check_url, args.provider)
        finally:
            store.close()
        return

    if args.scan or args.enqueue:
        try:
            enqueue_background_work(store)
        finally:
            store.close()
        return

    if args.worker:
        try:
            run_worker(
                store,
                once=args.once,
                stale_task_seconds=args.stale_task_seconds,
            )
        finally:
            store.close()
        return

    if args.render_cache:
        try:
            output_path = render_cached_report(store, args.output)
            if output_path and args.open:
                webbrowser.open(output_path.resolve().as_uri())
        finally:
            store.close()
        return

    if args.read or args.open:
        try:
            output_path = render_cached_report(store, args.output)
            if output_path:
                webbrowser.open(output_path.resolve().as_uri())
            else:
                print(
                    "No cached report yet. Run biasmeter --scan, then biasmeter --worker."
                )
        finally:
            store.close()
        return

    if args.embed_rss and not args.ingest_rss:
        try:
            embed_stored_rss_items(store, args.limit)
        finally:
            store.close()
        return

    if args.ingest_rss:
        rss_data = ingest_rss(store)

        if args.embed_rss:
            embed_stored_rss_items(store, args.limit)

        store.close()
        return

    # Fetch all RSS data
    rss_data = ingest_rss(store)
    rss_by_url = {item["link"]: item for item in rss_data}

    # Display fetched RSS data for verification

    """
    print("Fetched RSS data:")
    for item in rss_data:
        print(f"Title: {item['title']}")
        print(f"Description: {item['description']}\n")
    """
    results = group_rss_items_by_embedding(store, rss_data)

    if results:
        reports = []
        for group in results:
            topic = group
            articles = results[group]

            # Initialize an array to hold the content
            articles_content = []

            # Extract the content for each article based on its provider
            for article in articles:
                provider = article.get("provider")
                url = article.get("url")
                provider_config = providers.get(provider)
                selectors = (
                    provider_config.get("selectors", []) if provider_config else []
                )
                rss_item = rss_by_url.get(url, {})

                if provider and url and selectors:
                    try:
                        # Extract article content by provider-specific selector
                        article_content = extract_content_by_selector(
                            requests.get(
                                url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
                            ).text,
                            selectors,
                        )

                        if article_content:
                            article_document = {
                                "provider": provider,
                                "url": url,
                                "title": rss_item.get("title", ""),
                                "description": rss_item.get("description", ""),
                                "pub_date": rss_item.get("pub_date", ""),
                                "topic": topic,
                                "content": article_content,
                            }
                            article_document = enrich_article_with_summary_embedding(
                                store, article_document
                            )
                            article_document = store_article_revision(
                                store, article_document
                            )
                            articles_content.append(article_document)
                    except (
                        MistralRetryAfterError,
                        MistralRateLimitWithoutRetryAfterError,
                    ):
                        raise
                    except Exception as e:
                        print(f"Error extracting content for article {url}: {e}")

            # Send the extracted content to Mistral for comparison
            if articles_content:
                comparison_result = get_cached_topic_report(
                    store, topic, articles_content
                )

                if not comparison_result:
                    comparison_result = send_to_llm_for_comparison(
                        topic, articles_content
                    )

                if comparison_result:
                    comparison_result["topic"] = topic
                    comparison_result = store_topic_report_cache(
                        store, topic, articles_content, comparison_result
                    )
                    reports.append(comparison_result)
                    store.upsert("topic_report", topic, comparison_result)
                    build_topic_embedding(store, comparison_result, articles_content)
                    refresh_provider_bias_summary(store)

        if reports:
            output_path = render_html_report(reports, args.output)
            store.upsert(
                "html_report",
                str(output_path),
                {
                    "path": str(output_path),
                    "topic_count": len(reports),
                    "topics": [report.get("title") for report in reports],
                    "generated_at": datetime.now().isoformat(),
                },
            )
            print(f"\nReport written to {output_path}")

            if args.open:
                webbrowser.open(output_path.resolve().as_uri())
        else:
            print("No comparable article groups found.")
    else:
        print("No comparable article groups found.")

    store.close()


if __name__ == "__main__":
    main()
