"""The generated research write-up (§11.4, §16).

`research.md` is the tearsheet rendered as prose. It exists so a run can be
read without a UI, and so the honest caveats travel with the numbers rather
than living in a tooltip somebody has to hover over.

The tone is deliberate: it leads with what the run does **not** establish. A
generated report that opened with "Sharpe 0.71" and buried the deflated figure
three sections down would be doing the opposite of what this project is for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alphacore.report.tearsheet import Tearsheet

__all__ = ["RESEARCH_FILENAME", "render_research_markdown"]

RESEARCH_FILENAME = "research.md"

_WARNING_TEXT = {
    "UNCOSTED": (
        "**This backtest charged no costs.** Commission, spread and impact were all zero, "
        "so the equity curve below describes an execution nobody can obtain."
    ),
    "NO_BACKTEST": (
        "Economic metrics were not computed for this run, so the leaderboard is ranked on "
        "the information coefficient rather than net-of-cost Sharpe."
    ),
    "LOW_DSR": (
        "The deflated Sharpe ratio is low. After correcting for how many configurations "
        "were tried, this result is not distinguishable from the best of a set of "
        "coin flips."
    ),
    "SHORT_SAMPLE": (
        "The sample is short relative to the minimum track record length, so the Sharpe "
        "ratio has a wide confidence interval whatever its point estimate."
    ),
    "KILL_SWITCH": "The drawdown kill switch tripped and trading stopped before the end.",
    "UNSTOPPED_WARMUP": (
        "A position was held before the ATR warm-up completed, so the configured stops "
        "were not active for part of the run."
    ),
    "SUSPICIOUS_ACCURACY": (
        "Directional accuracy is above 0.55. Real short-horizon edges live at 0.51-0.54, "
        "so this usually indicates leakage rather than skill."
    ),
}


def _fmt(value: float | None, spec: str = ".3f", missing: str = "not computed") -> str:
    """Format a possibly-absent number.

    Args:
        value: The value.
        spec: Format spec.
        missing: Text to show when the value is None.

    Returns:
        The formatted string.
    """
    return missing if value is None else format(value, spec)


def render_research_markdown(tearsheet: Tearsheet) -> str:
    """Render a run as a markdown research note.

    Args:
        tearsheet: The completed tearsheet.

    Returns:
        Markdown. Deterministic given the tearsheet, so it can be regenerated
        and diffed (§4, I8).
    """
    headline = tearsheet.headline
    lines: list[str] = [
        f"# Research note — {tearsheet.run_id}",
        "",
        f"> {tearsheet.disclaimer}",
        "",
        "## What this run establishes",
        "",
    ]

    interval = headline["sharpe_ci"]
    interval_text = f"[{interval[0]:.3f}, {interval[1]:.3f}]" if interval else "not computed"

    lines += _verdict_lines(headline)
    lines += ["", "## Headline", "", "| measure | value |", "|---|---|"]
    lines += [
        f"| selected model | `{headline['model'] or 'none'}` |",
        f"| out-of-sample net Sharpe | {_fmt(headline['oos_net_sharpe'])} |",
        f"| Sharpe 95% interval | {interval_text} |",
        f"| deflated Sharpe (DSR) | {_fmt(headline['dsr'])} |",
        f"| probabilistic Sharpe (PSR) | {_fmt(headline['psr'])} |",
        f"| break-even round-trip cost | {_fmt(headline['break_even_bps'], '.1f')} bps |",
        f"| assumed round-trip cost | {_fmt(headline['assumed_round_trip_bps'], '.1f')} bps |",
        f"| probability of backtest overfitting | {_fmt(headline['pbo'], '.2f')} |",
        "",
    ]

    lines += _dataset_lines(tearsheet)
    lines += _validation_lines(tearsheet)
    lines += _leaderboard_lines(tearsheet)
    lines += _caveat_lines(tearsheet)

    lines += [
        "## Provenance",
        "",
        f"- config hash `{tearsheet.manifest.config_hash}`",
        f"- data hash `{tearsheet.manifest.data_hash or 'unrecorded'}`",
        f"- code version `{tearsheet.manifest.code_version or 'unrecorded'}`",
        f"- seed `{tearsheet.manifest.seed}`",
        f"- annualisation `{tearsheet.dataset.periods_per_year:.2f}` bars per year",
        "",
        f"_{tearsheet.disclaimer}_",
        "",
    ]
    return "\n".join(lines)


def _verdict_lines(headline: dict[str, Any]) -> list[str]:
    """The opening paragraph, which leads with the caveats.

    Args:
        headline: The tearsheet's headline mapping.

    Returns:
        Markdown lines.
    """
    sharpe = headline["oos_net_sharpe"]
    dsr = headline["dsr"]
    break_even = headline["break_even_bps"]
    assumed = headline["assumed_round_trip_bps"]

    if sharpe is None:
        return [
            "No costed backtest was run, so this note reports prediction quality only. "
            "Nothing here says whether the signal survives execution."
        ]

    lines = []
    if dsr is not None and dsr < 0.5:  # noqa: PLR2004
        lines.append(
            f"**This run does not establish an edge.** The out-of-sample Sharpe is "
            f"{sharpe:.3f}, but after deflating for the number of configurations tried "
            f"the probability that it is genuinely positive is {dsr:.2f}."
        )
    else:
        lines.append(
            f"The out-of-sample Sharpe is {sharpe:.3f}, and the deflated Sharpe ratio "
            f"of {_fmt(dsr, '.2f')} suggests it survives correction for the search."
        )

    if break_even is not None and assumed is not None:
        margin = break_even - assumed
        if margin <= 0:
            lines.append(
                f"The strategy breaks even at {break_even:.1f} bps of round-trip cost, "
                f"which is **below** the {assumed:.1f} bps assumed. It does not survive "
                "its own cost model."
            )
        else:
            lines.append(
                f"It breaks even at {break_even:.1f} bps against an assumed {assumed:.1f} "
                f"bps, leaving {margin:.1f} bps of margin."
            )
    return lines


def _dataset_lines(tearsheet: Tearsheet) -> list[str]:
    """The data section."""
    data = tearsheet.dataset
    lines = [
        "## Data",
        "",
        f"- symbols: {', '.join(data.symbols)}",
        f"- interval: `{data.interval}` on calendar `{data.calendar}`",
        f"- span: {data.span[0]} to {data.span[1]} ({data.n_bars:,} bars)",
        f"- quality: **{data.quality}**",
        "",
    ]
    if tearsheet.features:
        lines += [
            f"- features: {tearsheet.features.n_built} built, "
            f"{tearsheet.features.n_after_prune} after pruning, "
            f"{tearsheet.features.max_lookback} bars of warm-up dropped",
            "",
        ]
    if tearsheet.labels:
        labels = tearsheet.labels
        balance = ", ".join(f"`{k}` {v:.1%}" for k, v in sorted(labels.class_balance.items()))
        lines += [
            f"- labels: `{labels.method}` over {labels.horizon_bars} bars; class balance {balance}",
            "",
        ]
    return lines


def _validation_lines(tearsheet: Tearsheet) -> list[str]:
    """The cross-validation section."""
    if tearsheet.cv is None:
        return []
    cv = tearsheet.cv
    purged = sum(f.purged for f in cv.folds)
    embargoed = sum(f.embargoed for f in cv.folds)
    return [
        "## Validation",
        "",
        f"`{cv.scheme}` with {len(cv.folds)} folds and a {cv.embargo_pct:.1%} embargo. "
        f"Purging removed {purged:,} training samples whose label spans overlapped a test "
        f"window; the embargo removed a further {embargoed:,}.",
        "",
    ]


def _leaderboard_lines(tearsheet: Tearsheet) -> list[str]:
    """The leaderboard table, with the shuffled-label row called out."""
    if not tearsheet.leaderboard:
        return []
    lines = [
        "## Leaderboard",
        "",
        "| rank | model | net Sharpe | DSR | IC | IC t | AUC | flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in tearsheet.leaderboard:
        lines.append(
            f"| {row.rank} | `{row.model}` | {_fmt(row.oos_net_sharpe, '.3f', '—')} | "
            f"{_fmt(row.dsr, '.2f', '—')} | {_fmt(row.ic, '.4f', '—')} | "
            f"{_fmt(row.ic_tstat, '.2f', '—')} | {_fmt(row.auc, '.3f', '—')} | "
            f"{', '.join(row.flags) or '—'} |"
        )

    shuffled = next((r for r in tearsheet.leaderboard if r.model == "baseline_shuffled"), None)
    if shuffled is not None:
        lines += [
            "",
            f"The shuffled-label baseline — the same estimator trained on labels whose "
            f"relationship to the features has been destroyed — scored an IC of "
            f"{_fmt(shuffled.ic, '.4f', 'nothing measurable')}. Any model above it by less "
            "than that margin has not distinguished itself from noise.",
        ]
    lines.append("")
    return lines


def _caveat_lines(tearsheet: Tearsheet) -> list[str]:
    """The overfitting verdict and every warning, spelled out."""
    lines = ["## Caveats", "", tearsheet.overfitting.verdict, ""]
    if tearsheet.overfitting.n_configs_evaluated:
        lines += [
            f"{tearsheet.overfitting.n_configs_evaluated} configurations were evaluated "
            f"across {tearsheet.overfitting.n_splits} combinatorial splits.",
            "",
        ]
    if tearsheet.warnings:
        lines.append("The following flags were raised:")
        lines.append("")
        lines += [
            f"- **{flag}** — {_WARNING_TEXT.get(flag, 'See the run log for detail.')}"
            for flag in tearsheet.warnings
        ]
        lines.append("")
    return lines
