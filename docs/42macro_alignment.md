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
  --primary-strategy i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145
```

## Current Read: September 3, 2026

The refreshed public catalog contains 395 videos through September 3. Local
transcript coverage is 289 manifest-backed videos, and 278 of the 384 matched
comparisons are transcript-backed. Trade Bot's reconstructed operating history
now extends through September 3 under the configured primary strategy. The
remaining 106 matched rows are explicitly low-confidence title-only screens.

### Bottom line

The horizon-separated human read remains **constructive but fragile**, with
stronger duration and oil-tail warnings than on August 18. 42 Macro says its
current risk-on regime can survive one or two rate hikes, five of six macro
cycles remain tailwinds, and investors with horizons beyond the next few months
should buy volatility-driven dips. It simultaneously expects the next few
months to be volatile, sees the neutral policy rate rising, and treats eventual
yield-curve control plus an AI-capex secular bear market as material structural
risks. The September 3 BOJ discussion does not call a risk-off transition from
Japanese tightening alone.

Trade Bot is also approximately balanced rather than risk-off. On September 3
its risk score is 0.43, one-month risk-off probability is 23.38%, and final
defense is 48.57%. Its native base defense fell to 28.37%, but the independent
portfolio-risk layer added 20.20 percentage points, holding final defense at the
governed floor. Equity beta is 0.63 and worst modeled stress loss is 18.00%.
Scenario and narrative sizing authority remain zero.

The newest evidence therefore does not establish a current technology-cycle or
broad-market break. Both systems remain invested while watching whether higher
long yields, expensive AI financing, oil/refining pressure, and tighter funding
liquidity spread into credit, volatility, breadth, and trend. 42 Macro is more
explicitly bullish beyond the next few months; Trade Bot expresses more caution
through portfolio construction even though its native risk state is stable.

### New matched-date evidence

| Date | 42 Macro public message | Trade Bot matched state | Interpretation |
|---|---|---|---|
| Aug 24 | Treasury's policy bridge may need to lengthen; correction risk remains, while longer-horizon policy support is expected. | 48.57% defense; risk score 0.43; 1m risk-off 23.17%. | Horizon-mixed caution; the lexical score overstates an immediate sell call. |
| Aug 25 | The public videos split between a constructive policy view and concern that AI capital demand could accelerate fiscal stress. | 48.57% defense; risk score 0.43. | Mechanism agreement, but no unified tactical 42 Macro direction from the two videos. |
| Aug 27 | Fed policy and NVIDIA profitability were framed as a near-term risk question. | 48.57% defense; risk score 0.43. | The automated defensive score is a screening flag, not a verified allocation instruction. |
| Sep 1 | Run-it-hot policy is not expected to break bonds immediately; near-term volatility should be faded, while the eventual secular-bear warning remains. | 48.57% defense; risk score 0.43; 1m risk-off 23.38%. | Broad tactical alignment; the lexical classifier incorrectly labels the mixed-horizon transcript as risk reduction. |
| Sep 2 | The current risk-on regime can survive one or two hikes; five of six cycles are tailwinds, but the next few months may be volatile. Oil/refining pressure is a conditional inflation risk. | Same September 3 point-in-time match. | Constructive fragility and strong agreement on waiting for market confirmation of an oil shock. |
| Sep 3 | Faster BOJ hikes alone are unlikely to trigger a risk-off phase transition; tighter liquidity is still a headwind. A second video remains fundamentally bullish on risk assets. | 48.57% defense; equity beta 0.63; portfolio layer adds 20.20pp. | Both remain invested; Trade Bot carries the more defensive implementation. |

### Historical outcome comparison

On transcript-backed rows with realized forward windows, Trade Bot's allocation
proxy currently has the higher mean return at every tested horizon: 0.30% versus
0.22% at one week (273 rows), 0.98% versus 0.74% at one month (255 rows), and
2.16% versus 1.91% at three months (211 rows). Its mean action score is also
higher at each horizon. These are proxy allocations derived from public
language, not 42 Macro's proprietary portfolio, and the samples overlap heavily;
they support a descriptive comparison, not a claim of product superiority.

The mechanical classifier remains materially unreliable on mixed horizons. It
misreads the explicit dip-buying September 1 answer as defensive and compresses
the September 2 long-horizon bullish/near-term volatile view into one cautious
scalar. The human horizon read is authoritative for the current conclusion.

## Prior Read: August 18, 2026

The refreshed corpus contains 276 unique public videos through August 18. All
23 public videos published from July 20 through August 18 are transcript-backed,
and Trade Bot now has 22 business-day point-in-time rows over the same interval.
The history was rebuilt under one evaluation regime and with the current
configured primary strategy,
`i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145`.

### Bottom line

42 Macro's public stance changed in three phases. From July 28 through roughly
August 3 it emphasized scarce capital, hyperscaler credit, bond volatility,
possible Fed tightening, and the risk that an AI-financing unwind could become a
broader correction. From August 5 onward, Treasury financing policy and softer
labor/housing evidence reduced its perceived need for Fed tightening. By August
10 it answered that investors could remain sanguine "for now"; on August 11 it
called circular AI financing "overwhelmingly bullish for now" while retaining a
clear longer-run concentration and leverage warning. On August 14 it argued that
the AI capex bubble is likely to end in a secular bear market, but explicitly did
not call for immediate selling. On August 17-18 it identified capital competition
and rising global long yields as an accelerating correction risk while remaining
bullish over 12 months and cautious over the next few months.

Trade Bot completed the re-risking that was still pending on August 11. Final
defense fell from 66.37% on August 11 to 49.05% on August 14 and 48.57% on August
17-18. The August 18 risk score was 0.43 and the one-month risk-off probability
was 24.07%. Market trend, credit, AI leadership, and drawdown resilience remained
supportive; duration support (-0.26) and energy/inflation relief (-0.65) were
adverse, cross-sectional dispersion remained stressed, and the portfolio engine
added 2.51 percentage points of defense. Scenario and narrative sizing authority
remained zero.

The current comparison is therefore **broadly aligned constructive fragility**,
not the major stance divergence implied by the lexical score. 42 Macro is more
concerned about correction risk over the next few months. Trade Bot sees the same
duration/inflation pressure but does not yet see credit, volatility, or broad
trend confirmation of a systemic break.

### Matched-date evidence

| Date | 42 Macro public message | Trade Bot matched state | Interpretation |
|---|---|---|---|
| Jul 28 | Circular financing and the bond/funding channel increase correction risk, although 42 Macro did not say the cyclical bear had begun. | 53.22% defensive; risk score 0.40; 1m risk-off 25.25%; equity beta 0.93. | 42 Macro was more tactically cautious than the current Trade Bot strategy. |
| Jul 30 | The economy remained hot and the Goldilocks reaction looked too sanguine; Fed tightening risk was elevated. | 50.30% defensive; risk score 0.47; 1m risk-off 28.64%; equity beta 1.00. | Clear near-term divergence: the public macro warning strengthened while the strategy carried roughly neutral exposure. |
| Aug 4 | Six macro cycles were mostly positive, but monetary policy could become a headwind over the next one or two quarters; price signals were still required before action. | 54.02% defensive; risk score 0.53; 1m risk-off 30.94%; portfolio clamp added 3.81 points. | Broadly aligned, mixed/cautious states. |
| Aug 6 | Treasury's duration-scarcity financing could restrain bond vigilantes; 42 Macro remained bullish over 12 months and cautious over the next few months. | 53.70% defensive; risk score 0.50; 1m risk-off 28.80%; equity beta 1.05. | Similar near-term caution, although 42 Macro's policy interpretation was becoming more constructive. |
| Aug 7 | AI-driven cost control was described as bullish for corporate profits and equities, despite adverse social effects. | 66.76% defensive; risk score 0.50; 1m risk-off 28.73%; equity beta 0.62. | Major stance divergence. Trade Bot's jump came from its native selector/execution path, not a deteriorating risk score. |
| Aug 10 | Investors could remain sanguine "for now" because cooling labor/housing and policy support reduced the probability of Fed tightening. | 66.37% defensive; risk score 0.43; 1m risk-off 25.37%; equity beta 0.63. | 42 Macro was tactically constructive while Trade Bot remained mechanically defensive. |
| Aug 11 | Circular financing was "overwhelmingly bullish for now" because transferable compute collateral and institutional financing can extend the build; leverage and concentration remain the eventual bear case. | 66.37% defensive; risk score 0.40; 1m risk-off 22.67%; equity beta 0.62; stress loss 11.77%. | The strongest current divergence. Trade Bot's internals improved, but the implemented allocation still reflected the prior target through the execution lag. |
| Aug 14 | AI-capex overbuild, demanding EPS expectations, concentration, household exposure, and margin leverage create a high-probability eventual bear case; the current bull market was not declared over. | 49.05% defensive; risk score 0.33; 1m risk-off 19.90%; equity beta 0.63. | Broadly aligned tactically: remain invested while treating leverage and concentration as vulnerability, not a crash timer. |
| Aug 17 | Corporate AI financing is competing with sovereign duration and may require a wider equity risk premium in coming months. | 48.57% defensive; risk score 0.43; 1m risk-off 23.56%; duration support -0.26. | Trade Bot independently registered adverse duration conditions but retained a balanced allocation because credit and market trend remained supportive. |
| Aug 18 | Correction risk from capital competition was described as rapidly increasing; 42 Macro remained bullish over 12 months but warned that the next few months may be volatile. | 48.57% defensive; risk score 0.43; 1m risk-off 24.07%; portfolio clamp +2.51pp; equity beta 0.64. | Same broad regime, different emphasis: 42 Macro is more tactically cautious, while Trade Bot requires wider market confirmation before adding materially more defense. |

The correct synthesis is therefore:

- **Immediate posture:** both are constructive but fragile. 42 Macro is cautious
  over the next few months; Trade Bot is approximately balanced at 48.57%
  defense after completing its re-risk transition.
- **Structural agreement:** both still identify AI financing, leverage,
  concentration, sovereign competition for capital, and future Fed/bond-market
  interaction as important tail risks.
- **Confirmed break:** neither system currently establishes one. Trade Bot's
  credit and broad trend drivers are supportive, while 42 Macro calls correction
  risk rather than a current break and remains bullish over 12 months.
- **Highest-value monitor:** whether long-yield and expensive-growth weakness
  spreads into broad credit, volatility, correlation, breadth, and large-move
  participation. Capital-supply pressure is a mechanism; transmission into
  those markets is the actionable confirmation.

The mechanical lexical classifier still misreads the August 10 and 11 videos as
defensive because it counts their adverse branches more heavily than their
explicit near-term answers. On August 17-18 it gets the direction of the tactical
warning right but overstates the magnitude by collapsing a bullish 12-month view
and a cautious multi-month view into one scalar. The scalar remains a screening
proxy; the horizon-separated human read is authoritative for the current
comparison.

The agreement is not independent confirmation of every narrative claim. Trade
Bot's news/event and scenario-sizing authorities were zero on these matched
dates, so the allocation did not ingest the 42 Macro transcripts or the user's
narrative feed. The independent portfolio-stress engine did add 2.51 points of
defense by August 18. Both processes still respond to the same market and
economic system. The proper interpretation is independent construction with
overlapping underlying evidence, not statistically independent votes.

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
