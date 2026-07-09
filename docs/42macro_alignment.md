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

## Current Limitation

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
near-term posture from `-1` defensive to `+1` risk-on, then compares that score
to a trade-bot posture score derived from risk budget, risk score, and 1-month
risk-off probability.

Use the output as a disagreement audit, not a truth oracle. High-value rows are
the `large_change_focus` rows where either 42 Macro flagged a large tactical
shift or trade-bot's posture changed sharply.

Outcome scoring is also a tactical audit, not a complete test of 42 Macro's
long-horizon themes or proprietary model. Emphasize transcript-backed rows over
title-only rows.
