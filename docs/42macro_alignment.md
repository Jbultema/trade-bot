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

Compare saved 42 Macro transcripts to trade-bot operating history:

```bash
poetry run trade-bot compare-42macro
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
blocking indefinitely.

For older history, use one of these paths:

- Rerun the sync after a cooldown with a smaller `--max-videos` window.
- Use a browser transcript site manually and save transcript `.txt` files into
  `data/external/42macro_transcripts`.
- Keep the header format below so the compare command can match the transcript:

```text
source: 42macro_youtube
published_date: 2026-05-27
video_id: koWrV9ykBEo
url: https://www.youtube.com/watch?v=koWrV9ykBEo
title: The Macro Minute: Will ending the Strait of Hormuz Crisis be a sell-the-news catalyst?

<transcript text>
```

The compare command also scans local `.txt` files that are not yet present in
`manifest.json`, so manually added transcript files are picked up on the next
run.

## Interpretation

The classifier is keyword-weighted and transparent. It scores 42 Macro's
near-term posture from `-1` defensive to `+1` risk-on, then compares that score
to a trade-bot posture score derived from risk budget, risk score, and 1-month
risk-off probability.

Use the output as a disagreement audit, not a truth oracle. High-value rows are
the `large_change_focus` rows where either 42 Macro flagged a large tactical
shift or trade-bot's posture changed sharply.
