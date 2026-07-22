# 42 Macro Alignment Job

This workflow compares public 42 Macro YouTube commentary against trade-bot's
own point-in-time operating posture.

## What Gets Stored

- Transcripts and the video manifest live under `data/external/42macro_transcripts`.
- Comparison reports live under `reports/42macro_alignment`.
- Durable DuckDB tables:
  - `external_macro_videos`
  - `external_macro_classifications`
  - `external_macro_tradebot_comparisons`

The saved transcript files are local research artifacts. Do not paste full
third-party transcripts into reports or app copy.

## Permission Boundary

This workflow does not grant permission to scrape, retain, benchmark, or reuse
third-party content. 42 Macro's current published terms contain broad
restrictions on automated collection, systematic analysis/backtesting, model
development, and redistribution. Before running transcript automation, confirm
that the specific public content and intended personal use are permitted; use
human-authored notes or link-only metadata when permission is uncertain. Never
apply this workflow to gated or subscriber content without explicit rights.

## Commands

Sync public video metadata and transcript text:

```bash
poetry run trade-bot sync-42macro-transcripts --max-videos 300 --max-pages 30
```

Write a prioritized queue of missing transcripts:

```bash
poetry run trade-bot prioritize-42macro-transcripts
```

The queue is written to:

- `reports/42macro_alignment/missing_transcript_priority.csv`

It includes both the normal YouTube URL and a
`https://youtubetotranscript.com/transcript?v=...` URL so high-value missing
transcripts can be opened directly in a browser.

Import browser-copied transcripts or saved transcript pages:

```bash
poetry run trade-bot import-42macro-transcripts --input-dir data/external/42macro_manual_imports
```

Compare saved 42 Macro transcripts to trade-bot operating history:

```bash
poetry run trade-bot compare-42macro
```

Score both systems against what happened next:

```bash
poetry run trade-bot score-42macro-outcomes
```

Daily job wrapper:

```bash
poetry run trade-bot run-42macro-daily-check
```

If comparison coverage is sparse, backfill trade-bot operating rows first:

```bash
poetry run trade-bot seed-operating-history \
  --source latest-snapshot \
  --frequency B \
  --daily-tail-market-days 0 \
  --max-points 2000 \
  --primary-strategy i111_reentry_vol_target_fast_21d
```

## Current Read: July 21, 2026

The refreshed corpus includes 256 transcript-backed videos through July 21,
including the July 20 AI-capex pair and the July 21 short-squeeze/correction
discussion. The current comparison is broadly aligned after correcting an old
posture-mapping error:

- 42 Macro's July 21 transcript is classified `constructive_but_fragile`
  (-0.15). Its horizon-specific message is more nuanced than that scalar: the
  current regime is still risk-on, crowded shorts can support a squeeze first,
  and a correction, recovery, and eventual crash/regime break remain plausible
  in that order.
- Trade Bot is `cautious` (-0.27), based on its actual 63.73% final defensive
  allocation rather than its 90% risk-budget capacity.
- The absolute posture gap is 0.13, which is an aligned classification. The
  systems agree most clearly on elevated dispersion/concentration, AI-capex
  fragility, and credit as a confirmation channel. Trade Bot is more defensive
  on immediate sizing; 42 Macro is more explicit that positioning can insulate
  the index and power a near-term squeeze before the regime changes.

The six newest transcript-backed videos add useful horizon detail that the
single lexical score cannot preserve:

| Date | Public topic | Human horizon read | Comparison with Trade Bot |
|---|---|---|---|
| Jul 14 | Fed tightening and AI-capex bubble | Fed likely on hold, but AI spending and concentration are bubble-like; eventual easing is medium-term support. | Same fragility concern; 42 Macro is more constructive about the medium-term policy path. |
| Jul 15 | K-shaped monetary policy | A structural policy transition is a downside risk to gold, Bitcoin, and, to a lesser degree, stocks; potentially bullish for Treasuries. | Mixed cross-asset view, not a clean equity risk-on call. The lexical `+1.00` score is an obvious classifier overstatement. |
| Jul 16 | Warsh Fed winners and losers | Mixed: a more inflation-sensitive Fed is a risk to risk assets, while realtime inflation measures and AI productivity can support dovish policy. | Directionally compatible with caution, but not directly comparable to a BIL-versus-risk allocation. |
| Jul 20 | China and hyperscaler capex | Highest-to-date risk of a capex reset, with no timing claim; AI leaders can weaken while the broad index remains resilient. | Strong thematic alignment with Trade Bot's concentration/dispersion warning. |
| Jul 20 | The U.S.-China AI-race narrative | Explicitly skeptical of the financing and investment case around concentrated U.S. AI spending. | Strong thematic alignment, but this is narrative evidence rather than an independent sizing input. |
| Jul 21 | Squeeze, correction, recovery, crash | Present regime remains risk-on; crowded shorts favor a squeeze first, then a correction and recovery, with a later regime break/crash risk. Credit is neutral rather than confirming a broad break. | Broadly aligned on fragility and lack of break confirmation; Trade Bot is materially more defensive today. |

The cleanest synthesis is therefore horizon-dependent. Over days to a few
weeks, 42 Macro is more constructive because it expects positioning to support
the index. Over the next one to two quarters, both processes favor caution
around Fed uncertainty, AI-capex deceleration, dispersion, concentration, and
credit confirmation. Over the longer run, 42 Macro remains constructive on AI
diffusion while warning that the capex bubble can eventually end in a secular
bear phase; Trade Bot does not have credible authority at that horizon.

The agreement is not independent confirmation of every narrative claim. Trade
Bot's current news/event, scenario-sizing, and scenario-conditioned portfolio
authorities are all zero, so its *allocation* is independent of the 42 Macro
transcripts and the user's narrative feed. Its base strategy and quantitative
risk-status layer still use market prices that respond to the same underlying
world. The correct interpretation is independent causal construction with
overlapping market evidence, not statistically independent votes.

Across matured transcript-backed observations, Trade Bot's defensive sizing
scored better on the report's action metric (0.60/0.72/0.66 at 1w/1m/3m versus
0.48/0.45/0.42 for the transcript proxy), while 42 Macro's return proxy was
higher at 1m and 3m. This is a simplified tactical audit, not proof of
superiority: the text classifier compresses horizon-specific commentary into a
single scalar, and the 42 Macro proxy is not their proprietary portfolio.

## Current Limitations

The public channel catalog can be discovered from YouTube, but caption endpoints
can return HTTP 429 or IP-block errors when many older transcripts are requested
in one run. The sync command records those failures in the manifest instead of
blocking indefinitely. Browser transcript sites can show text to a normal
interactive browser while returning bot-challenge HTML to direct scripts, so the
manual import path is the preferred fallback when transcript coverage matters.

For older history, use one of these paths:

- Rerun the sync after a cooldown with a smaller `--max-videos` window.
- Run `prioritize-42macro-transcripts`, open the highest-priority
  `transcript_url` rows, copy the visible transcript into `.txt` files under
  `data/external/42macro_manual_imports`, then run `import-42macro-transcripts`.
- Save transcript pages as `.html` files into `data/external/42macro_manual_imports`
  and import them the same way.
- If you are creating transcript files by hand, the header format below is still
  accepted and gives the importer an exact match:

```text
source: 42macro_youtube
published_date: 2026-05-27
video_id: koWrV9ykBEo
url: https://www.youtube.com/watch?v=koWrV9ykBEo
title: The Macro Minute: Will ending the Strait of Hormuz Crisis be a sell-the-news catalyst?

<transcript text>
```

The importer can infer the video from a file name containing the YouTube ID, a
YouTube or youtubetotranscript URL in the file, the header above, or a title that
matches the manifest.

## Forward Outcome Scoring

The outcome job asks a different question from the alignment report:

- Alignment: did 42 Macro and trade-bot say roughly the same thing?
- Outcome scoring: was each system's tactical risk sizing appropriate for what
  happened afterward?

The current implementation tests 1-week, 1-month, and 3-month forward windows.
It converts each system's posture to a risk-allocation proxy, then compares the
proxy against SPY versus BIL outcomes:

- Constructive windows reward higher risk exposure.
- Left-tail return or drawdown windows reward lower risk exposure.
- Choppy windows reward balanced sizing.

Output files:

- `reports/42macro_alignment/forward_outcome_scores.csv`
- `reports/42macro_alignment/forward_outcome_summary.csv`
- `reports/42macro_alignment/outcome_analysis.md`

## Interpretation

The classifier is keyword-weighted and transparent. It scores 42 Macro's
near-term posture from `-1` defensive to `+1` risk-on. Trade Bot posture is
derived directly from final defensive allocation when available:
`posture = 1 - 2 * final_defensive_weight`. Older rows without that field fall
back to risk budget, risk score, and one-month risk-off probability. The direct
allocation mapping is essential because risk-budget capacity is not total
exposure; on July 21 the capacity multiplier was 0.90 while the final portfolio
was 63.73% defensive.

The transcript score is explicitly a lexical proxy. Repetition, negation,
quoted community questions, and mixed tactical/structural horizons can distort
it. The July 15 `risk_on` classification is the current clearest failure case.
Use the recent-video horizon read above for the current conclusion; retain the
mechanical scalar only for repeatable historical screening and outcome audits.

Use the output as a disagreement audit, not a truth oracle. High-value rows are
the `large_change_focus` rows where either 42 Macro flagged a large tactical
shift or trade-bot's posture changed sharply.

Outcome scoring is also a tactical audit, not a complete test of 42 Macro's
long-horizon themes or proprietary model. Emphasize transcript-backed rows over
title-only rows.
