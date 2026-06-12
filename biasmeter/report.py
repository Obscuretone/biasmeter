import html
from datetime import datetime
from pathlib import Path

PATTERN_CATEGORIES = {
    "inclusions": "Unique inclusions",
    "omissions": "Omissions",
    "framing_differences": "Framing differences",
    "bias_signals": "Bias signals",
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
    }


def add_pattern_example(pattern, category, topic, detail, context=""):
    pattern["counts"][category] += 1

    if len(pattern["examples"][category]) >= 5:
        return

    pattern["examples"][category].append(
        {
            "topic": topic,
            "detail": detail,
            "context": context,
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
            )

        for item in report.get("provider_specific_omissions", []):
            provider = item.get("provider", "unknown")
            add_pattern_example(
                pattern_for(provider),
                "omissions",
                topic,
                item.get("missing_detail", ""),
                "Covered by: " + ", ".join(item.get("covered_by", [])),
            )

        for item in report.get("framing_differences", []):
            for provider in item.get("providers", []):
                add_pattern_example(
                    pattern_for(provider),
                    "framing_differences",
                    topic,
                    item.get("difference", ""),
                )

        for item in report.get("potential_bias_signals", []):
            for provider in item.get("providers", []):
                add_pattern_example(
                    pattern_for(provider),
                    "bias_signals",
                    topic,
                    item.get("signal", ""),
                    f"{item.get('confidence', 'unknown')} confidence",
                )

    return {
        provider: pattern
        for provider, pattern in sorted(patterns.items())
        if any(pattern["counts"].values())
    }


def format_pattern_example(example):
    topic = html.escape(example.get("topic", "Untitled topic"))
    detail = html.escape(example.get("detail", ""))
    context = html.escape(example.get("context", ""))
    context_html = f'<br><span class="muted">{context}</span>' if context else ""
    return f"<strong>{topic}</strong>: {detail}{context_html}"


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
                <div class="pattern-chips">{chips}</div>
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


def render_html_report(reports, output_path):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    topic_sections = []
    provider_patterns = build_provider_patterns(reports)

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

                <h3>Provider-Specific Inclusions</h3>
                {render_list_items(report.get("provider_specific_inclusions", []), format_provider_inclusion)}

                <h3>Provider-Specific Omissions</h3>
                {render_list_items(report.get("provider_specific_omissions", []), format_provider_omission)}

                <h3>Framing Differences</h3>
                {render_list_items(report.get("framing_differences", []), format_framing_difference)}

                <h3>Potential Bias Signals</h3>
                {render_list_items(report.get("potential_bias_signals", []), format_bias_signal)}

                <p class="confidence">Overall confidence: {html.escape(report.get("confidence", "unknown"))}</p>
            </section>
            """
        )

    document = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Biasmeter Coverage Report</title>
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
            <h1>Biasmeter Coverage Report</h1>
            <p class="muted">Generated {html.escape(generated_at)}. Hover over highlighted sentences to see which provider each sentence came from.</p>
        </header>
        <main>
            {render_provider_patterns(provider_patterns)}
            {"".join(topic_sections)}
        </main>
    </body>
    </html>
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
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
    print(f"Cached report written to {output_path}")
    return output_path
