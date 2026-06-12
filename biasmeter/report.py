import html
import re
from datetime import datetime
from pathlib import Path

PATTERN_CATEGORIES = {
    "inclusions": "Unique inclusions",
    "omissions": "Omissions",
    "framing_differences": "Framing differences",
    "bias_signals": "Bias signals",
}

INFORMATION_TYPE_KEYWORDS = {
    "public safety and enforcement": (
        "police",
        "arrest",
        "crime",
        "safety",
        "enforcement",
        "investigation",
        "collision",
        "crash",
        "fire",
        "emergency",
        "spvm",
        "sq",
    ),
    "government accountability": (
        "government",
        "minister",
        "audit",
        "auditor",
        "report",
        "public money",
        "policy",
        "bill",
        "law",
        "legislation",
        "accountability",
    ),
    "costs, taxes, and affordability": (
        "cost",
        "price",
        "tax",
        "qst",
        "rent",
        "fare",
        "fee",
        "funding",
        "money",
        "budget",
        "affordability",
    ),
    "health and consumer risk": (
        "health",
        "recall",
        "food",
        "milk",
        "drug",
        "hospital",
        "disease",
        "risk",
        "warning",
        "ban",
    ),
    "transit and infrastructure": (
        "transit",
        "rem",
        "stm",
        "metro",
        "bus",
        "road",
        "bridge",
        "parking",
        "station",
        "traffic",
        "infrastructure",
    ),
    "weather and environment": (
        "weather",
        "storm",
        "flood",
        "rain",
        "heat",
        "cold",
        "water",
        "environment",
        "climate",
        "forecast",
    ),
    "who is quoted or centered": (
        "quote",
        "quoted",
        "interview",
        "said",
        "voice",
        "residents",
        "officials",
        "advocates",
        "experts",
        "critics",
    ),
    "context and background": (
        "context",
        "background",
        "history",
        "previous",
        "data",
        "statistics",
        "comparison",
        "timeline",
    ),
}


def tooltip_for_sources(sources):
    source_lines = []
    for source in sources:
        provider = source.get("provider", "unknown")
        support = source.get("support", "")
        source_lines.append(f"{provider}: {support}")

    return "\n".join(source_lines)


def render_list_items(items, formatter):
    if not items:
        return '<p class="muted">No items reported.</p>'

    return "<ul>" + "".join(f"<li>{formatter(item)}</li>" for item in items) + "</ul>"


def format_cliffs_note(item):
    return html.escape(str(item))


def format_provider_inclusion(item):
    provider = html.escape(item.get("provider", "unknown"))
    detail = html.escape(item.get("detail", ""))
    absent_from = html.escape(", ".join(item.get("absent_from", [])))
    why_it_matters = html.escape(item.get("why_it_matters", ""))
    return (
        f"<strong>{provider}</strong>: {detail} "
        f'<span class="muted">Absent from: {absent_from}</span><br>'
        f"{why_it_matters}"
    )


def format_provider_omission(item):
    provider = html.escape(item.get("provider", "unknown"))
    missing_detail = html.escape(item.get("missing_detail", ""))
    covered_by = html.escape(", ".join(item.get("covered_by", [])))
    why_it_matters = html.escape(item.get("why_it_matters", ""))
    return (
        f"<strong>{provider}</strong>: {missing_detail} "
        f'<span class="muted">Covered by: {covered_by}</span><br>'
        f"{why_it_matters}"
    )


def format_framing_difference(item):
    providers = html.escape(", ".join(item.get("providers", [])))
    difference = html.escape(item.get("difference", ""))
    return f"<strong>{providers}</strong>: {difference}"


def format_bias_signal(item):
    confidence = html.escape(item.get("confidence", "unknown"))
    signal = html.escape(item.get("signal", ""))
    evidence = html.escape(item.get("evidence", ""))
    return (
        f"<strong>{confidence} confidence</strong>: {signal}<br>"
        f'<span class="muted">{evidence}</span>'
    )


def new_provider_pattern():
    return {
        "counts": {category: 0 for category in PATTERN_CATEGORIES},
        "examples": {category: [] for category in PATTERN_CATEGORIES},
        "themes": {category: {} for category in PATTERN_CATEGORIES},
        "topics": set(),
    }


def normalize_theme_label(value):
    value = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    return value or "uncategorized detail"


def infer_information_type(*parts):
    text = " ".join(str(part or "") for part in parts).lower()
    for label, keywords in INFORMATION_TYPE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return label

    return "other recurring detail"


def add_theme(pattern, category, theme):
    theme = normalize_theme_label(theme)
    category_themes = pattern["themes"][category]
    category_themes[theme] = category_themes.get(theme, 0) + 1


def add_pattern_example(pattern, category, topic, detail, context="", theme=None):
    pattern["counts"][category] += 1
    pattern["topics"].add(topic)
    theme = theme or infer_information_type(detail, context)
    add_theme(pattern, category, theme)

    if len(pattern["examples"][category]) >= 5:
        return

    pattern["examples"][category].append(
        {
            "topic": topic,
            "detail": detail,
            "context": context,
            "theme": theme,
        }
    )


def build_provider_patterns(reports):
    patterns = {}

    def pattern_for(provider):
        provider = provider or "unknown"
        if provider not in patterns:
            patterns[provider] = new_provider_pattern()
        return patterns[provider]

    for report in reports:
        topic = report.get("title") or report.get("topic") or "Untitled topic"

        for item in report.get("provider_specific_inclusions", []):
            provider = item.get("provider", "unknown")
            add_pattern_example(
                pattern_for(provider),
                "inclusions",
                topic,
                item.get("detail", ""),
                "Absent from: " + ", ".join(item.get("absent_from", [])),
                item.get("information_type"),
            )

        for item in report.get("provider_specific_omissions", []):
            provider = item.get("provider", "unknown")
            add_pattern_example(
                pattern_for(provider),
                "omissions",
                topic,
                item.get("missing_detail", ""),
                "Covered by: " + ", ".join(item.get("covered_by", [])),
                item.get("information_type"),
            )

        for item in report.get("framing_differences", []):
            for provider in item.get("providers", []):
                add_pattern_example(
                    pattern_for(provider),
                    "framing_differences",
                    topic,
                    item.get("difference", ""),
                    "",
                    item.get("framing_axis"),
                )

        for item in report.get("potential_bias_signals", []):
            for provider in item.get("providers", []):
                add_pattern_example(
                    pattern_for(provider),
                    "bias_signals",
                    topic,
                    item.get("signal", ""),
                    f"{item.get('confidence', 'unknown')} confidence",
                    item.get("bias_axis"),
                )

    return {
        provider: finalize_provider_pattern(pattern)
        for provider, pattern in sorted(patterns.items())
        if any(pattern["counts"].values())
    }


def strongest_provider_categories(pattern):
    counts = pattern["counts"]
    return [
        category
        for category, count in sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if count
    ]


def strongest_provider_themes(pattern, limit=4):
    themes = []
    for category, category_themes in pattern["themes"].items():
        for theme, count in category_themes.items():
            themes.append((theme, category, count))

    return sorted(themes, key=lambda item: item[2], reverse=True)[:limit]


def describe_theme(theme, category, count):
    category_label = PATTERN_CATEGORIES[category].lower()
    return f"{theme} ({category_label}, {count})"


def summarize_provider_pattern(pattern):
    observed_topic_count = len(pattern["topics"])
    counts = pattern["counts"]
    total_signals = sum(counts.values())

    if total_signals == 0:
        return "No recurring discrepancy signals have been found for this provider yet."

    strongest_categories = strongest_provider_categories(pattern)
    strongest_labels = [
        PATTERN_CATEGORIES[category].lower() for category in strongest_categories[:2]
    ]
    if len(strongest_labels) == 1:
        strongest_phrase = strongest_labels[0]
    else:
        strongest_phrase = f"{strongest_labels[0]} and {strongest_labels[1]}"

    consistency_note = (
        "This is an early signal, not yet a consistent pattern."
        if observed_topic_count < 2
        else "This appears across multiple cached topics and is worth watching over time."
    )
    bias_signal_note = ""
    if counts["bias_signals"]:
        bias_signal_note = (
            f" {counts['bias_signals']} cautious bias signal(s) were flagged."
        )
    recurring_themes = [
        describe_theme(theme, category, count)
        for theme, category, count in strongest_provider_themes(pattern, limit=3)
        if count > 1
    ]
    recurring_theme_note = ""
    if recurring_themes:
        recurring_theme_note = (
            " Repeated information types include " + ", ".join(recurring_themes) + "."
        )

    return (
        f"Across {observed_topic_count} cached topic(s), this provider has "
        f"{total_signals} discrepancy signal(s), mostly {strongest_phrase}."
        f"{bias_signal_note}{recurring_theme_note} {consistency_note}"
    )


def finalize_provider_pattern(pattern):
    finalized_pattern = {
        "counts": pattern["counts"],
        "examples": pattern["examples"],
        "themes": pattern["themes"],
        "strongest_themes": [
            {
                "theme": theme,
                "category": category,
                "count": count,
            }
            for theme, category, count in strongest_provider_themes(pattern)
        ],
        "observed_topic_count": len(pattern["topics"]),
        "topics": sorted(pattern["topics"]),
    }
    finalized_pattern["summary"] = summarize_provider_pattern(pattern)
    return finalized_pattern


def format_pattern_example(example):
    topic = html.escape(example.get("topic", "Untitled topic"))
    detail = html.escape(example.get("detail", ""))
    context = html.escape(example.get("context", ""))
    theme = html.escape(example.get("theme", ""))
    context_html = f'<br><span class="muted">{context}</span>' if context else ""
    theme_html = f'<br><span class="muted">Type: {theme}</span>' if theme else ""
    return f"<strong>{topic}</strong>: {detail}{context_html}{theme_html}"


def render_provider_themes(pattern):
    themes = pattern.get("strongest_themes", [])
    if not themes:
        return '<p class="muted">No recurring information types detected yet.</p>'

    items = []
    for theme in themes:
        items.append(
            "<li>"
            f"<strong>{html.escape(theme.get('theme', 'unknown'))}</strong>: "
            f"{html.escape(PATTERN_CATEGORIES.get(theme.get('category'), 'signals').lower())}, "
            f"{theme.get('count', 0)} signal(s)"
            "</li>"
        )

    return "<ul>" + "".join(items) + "</ul>"


def render_provider_patterns(patterns):
    if not patterns:
        return """
        <section class="topic provider-patterns">
            <h2>Provider Patterns</h2>
            <p class="muted">No repeated provider-level discrepancy patterns are cached yet.</p>
        </section>
        """

    provider_sections = []
    for provider, pattern in patterns.items():
        counts = pattern["counts"]
        chips = " ".join(
            f'<span class="pattern-chip">{html.escape(label)}: {counts[category]}</span>'
            for category, label in PATTERN_CATEGORIES.items()
            if counts[category]
        )
        example_sections = []
        for category, label in PATTERN_CATEGORIES.items():
            examples = pattern["examples"][category]
            if not examples:
                continue

            example_sections.append(
                f"""
                <h4>{html.escape(label)}</h4>
                {render_list_items(examples, format_pattern_example)}
                """
            )

        provider_sections.append(
            f"""
            <article class="provider-pattern">
                <h3>{html.escape(provider)}</h3>
                <p>{html.escape(pattern.get("summary", ""))}</p>
                <p class="muted">Observed across {pattern.get("observed_topic_count", 0)} cached topic(s).</p>
                <div class="pattern-chips">{chips}</div>
                <h4>Recurring Information Types</h4>
                {render_provider_themes(pattern)}
                {"".join(example_sections)}
            </article>
            """
        )

    return f"""
    <section class="topic provider-patterns">
        <h2>Provider Patterns</h2>
        <p class="muted">Cumulative signals across cached topics. These are evidence trails for review, not proof of intent.</p>
        {"".join(provider_sections)}
    </section>
    """


def provider_breakdowns_path(output_path):
    output_path = Path(output_path)
    return output_path.with_name("provider-breakdowns.html")


def render_topic_sections(reports, include_analysis=True):
    topic_sections = []
    for report in reports:
        title = html.escape(
            report.get("title") or report.get("topic") or "Untitled topic"
        )
        cliffs_notes = report.get("cliffs_notes", [])
        full_text = report.get("full_text", [])

        sourced_sentences = []
        for entry in full_text:
            sentence = html.escape(entry.get("sentence", ""))
            sources = entry.get("sources", [])
            tooltip = html.escape(tooltip_for_sources(sources), quote=True)
            providers = ", ".join(
                sorted({source.get("provider", "unknown") for source in sources})
            )
            sourced_sentences.append(
                f'<span class="sourced-sentence" title="{tooltip}">'
                f"{sentence}"
                f"<sup>{html.escape(providers)}</sup>"
                "</span>"
            )

        source_links = {}
        for entry in full_text:
            for source in entry.get("sources", []):
                provider = source.get("provider", "unknown")
                url = source.get("url")
                if url:
                    source_links.setdefault(provider, set()).add(url)

        source_link_html = "".join(
            f"<li><strong>{html.escape(provider)}</strong>: "
            + ", ".join(
                f'<a href="{html.escape(url, quote=True)}">{html.escape(url)}</a>'
                for url in sorted(urls)
            )
            + "</li>"
            for provider, urls in sorted(source_links.items())
        )

        analysis_html = ""
        if include_analysis:
            analysis_html = f"""
                <h3>Provider-Specific Inclusions</h3>
                {render_list_items(report.get("provider_specific_inclusions", []), format_provider_inclusion)}

                <h3>Provider-Specific Omissions</h3>
                {render_list_items(report.get("provider_specific_omissions", []), format_provider_omission)}

                <h3>Framing Differences</h3>
                {render_list_items(report.get("framing_differences", []), format_framing_difference)}

                <h3>Potential Bias Signals</h3>
                {render_list_items(report.get("potential_bias_signals", []), format_bias_signal)}

                <p class="confidence">Overall confidence: {html.escape(report.get("confidence", "unknown"))}</p>
            """

        topic_sections.append(
            f"""
            <section class="topic">
                <h2>{title}</h2>

                <div class="cliffs-notes">
                    <h3>Cliffs Notes</h3>
                    {render_list_items(cliffs_notes, format_cliffs_note)}
                </div>

                <details>
                    <summary>Full sourced text</summary>
                    <div class="coverage-text">{" ".join(sourced_sentences)}</div>
                </details>

                <details open>
                    <summary>Source links</summary>
                    <ul>{source_link_html}</ul>
                </details>

                {analysis_html}
            </section>
            """
        )

    return "".join(topic_sections)


def render_page(title, subtitle, active_page, body_html, output_path, recent_path):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    recent_class = "active" if active_page == "recent" else ""
    providers_class = "active" if active_page == "providers" else ""
    output_path = Path(output_path)
    recent_href = html.escape(
        Path(recent_path).name if active_page == "providers" else "#"
    )
    providers_href = html.escape(
        provider_breakdowns_path(recent_path).name if active_page == "recent" else "#"
    )

    document = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{html.escape(title)}</title>
        <style>
            :root {{
                color-scheme: light dark;
                --bg: #101216;
                --card: #181c23;
                --text: #f0f2f5;
                --muted: #aab3c0;
                --accent: #ffd166;
                --border: #2c3340;
            }}
            body {{
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: var(--bg);
                color: var(--text);
                line-height: 1.6;
            }}
            header, main {{
                max-width: 980px;
                margin: 0 auto;
                padding: 24px;
            }}
            nav {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 18px;
            }}
            nav a {{
                border: 1px solid var(--border);
                border-radius: 999px;
                color: var(--text);
                font-weight: 700;
                padding: 7px 14px;
                text-decoration: none;
            }}
            nav a.active {{
                background: rgba(255, 209, 102, 0.14);
                border-color: rgba(255, 209, 102, 0.45);
                color: var(--accent);
            }}
            .topic {{
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 18px;
                margin: 24px 0;
                padding: 24px;
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
            }}
            .coverage-text {{
                font-size: 1.1rem;
                margin: 18px 0 24px;
            }}
            .cliffs-notes {{
                background: rgba(255, 209, 102, 0.08);
                border: 1px solid rgba(255, 209, 102, 0.25);
                border-radius: 14px;
                margin: 18px 0 24px;
                padding: 16px 18px;
            }}
            .cliffs-notes h3 {{
                margin-top: 0;
            }}
            .cliffs-notes li {{
                margin: 8px 0;
            }}
            .sourced-sentence {{
                background: rgba(255, 209, 102, 0.14);
                border-bottom: 2px solid var(--accent);
                border-radius: 4px;
                cursor: help;
                padding: 1px 3px;
            }}
            sup {{
                color: var(--accent);
                font-size: 0.65rem;
                margin-left: 3px;
            }}
            a {{
                color: var(--accent);
            }}
            details {{
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 12px 16px;
            }}
            summary {{
                cursor: pointer;
                font-weight: 700;
            }}
            .muted {{
                color: var(--muted);
            }}
            .confidence {{
                color: var(--muted);
                font-weight: 700;
            }}
            .provider-pattern {{
                border-top: 1px solid var(--border);
                margin-top: 18px;
                padding-top: 14px;
            }}
            .pattern-chips {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin: 10px 0 14px;
            }}
            .pattern-chip {{
                background: rgba(255, 209, 102, 0.12);
                border: 1px solid rgba(255, 209, 102, 0.28);
                border-radius: 999px;
                color: var(--accent);
                font-size: 0.85rem;
                font-weight: 700;
                padding: 3px 10px;
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>{html.escape(title)}</h1>
            <p class="muted">{html.escape(subtitle)}</p>
            <p class="muted">Generated {html.escape(generated_at)}. Hover over highlighted sentences to see which provider each sentence came from.</p>
            <nav aria-label="Biasmeter pages">
                <a class="{recent_class}" href="{recent_href}">Recent News</a>
                <a class="{providers_class}" href="{providers_href}">Provider Breakdowns</a>
            </nav>
        </header>
        <main>
            {body_html}
        </main>
    </body>
    </html>
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def render_html_report(reports, output_path):
    output_path = Path(output_path)
    provider_path = provider_breakdowns_path(output_path)
    provider_patterns = build_provider_patterns(reports)

    render_page(
        "Biasmeter Recent News",
        "A cache-first news reader with concise factual notes and sourced full text.",
        "recent",
        render_topic_sections(reports, include_analysis=False),
        output_path,
        output_path,
    )
    render_page(
        "Biasmeter Provider Breakdowns",
        "Long-term inclusion, omission, framing, and bias-signal patterns across cached coverage.",
        "providers",
        render_provider_patterns(provider_patterns)
        + render_topic_sections(reports, include_analysis=True),
        provider_path,
        output_path,
    )
    return output_path


def render_cached_report(store, output_path):
    reports = [row["content"] for row in store.list_by_type("topic_report")]
    provider_patterns = build_provider_patterns(reports)

    if not reports:
        print("No cached topic reports found yet.")
        return None

    output_path = render_html_report(reports, output_path)
    store.upsert(
        "html_report",
        str(output_path),
        {
            "path": str(output_path),
            "provider_breakdowns_path": str(provider_breakdowns_path(output_path)),
            "paths": {
                "recent_news": str(output_path),
                "provider_breakdowns": str(provider_breakdowns_path(output_path)),
            },
            "topic_count": len(reports),
            "topics": [report.get("title") for report in reports],
            "generated_at": datetime.now().isoformat(),
            "source": "cache",
        },
    )
    store.upsert(
        "provider_bias_summary",
        "latest",
        {
            "generated_at": datetime.now().isoformat(),
            "topic_count": len(reports),
            "providers": provider_patterns,
        },
    )
    print(f"Recent news written to {output_path}")
    print(f"Provider breakdowns written to {provider_breakdowns_path(output_path)}")
    return output_path
