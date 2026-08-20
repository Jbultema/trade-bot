# Capital-Cycle Break Monitor

Status: V0/V1 research explainer. Allocation authority: **0%**. Last reviewed: 2026-08-19.

## Purpose

The Capital-Cycle Break Monitor extends Cycle Tracker upstream from market recognition. It asks whether a capital-intensive buildout is merely fragile or whether the economic and financing mechanisms capable of ending it have begun to roll over. AI is the first application; the intended research object is the general capital cycle.

It is not an AI bubble score, crash forecast, sizing overlay, or a second copy of
Trade-bot's existing market-risk engine.

## Module boundary

The monitor owns only evidence that can add information about whether a physical,
contractual, financing, or monetization cycle is beginning to break:

- capital and capacity commitments;
- orders, backlog, cancellations, and capex revisions;
- utilization and pricing;
- project-specific financing availability and terms;
- customer monetization, margins, cash burn, and payback;
- the ordered transition from vulnerability to real-economy rollover.

Trend, breadth, volatility, broad credit, rates, leverage, and market concentration
already belong to Trade-bot. They may be displayed as frozen comparators or used as
downstream confirmation, but they do not enter the upstream vulnerability or rollover
composites. A proxy-only rearrangement of those signals is a negative control, not a
candidate new model.

## Two separate questions

The monitor reports two composites rather than collapsing them into one bearish score:

- **Capital vulnerability:** accumulated commitments, financing dependence, weak monetization, concentration, leverage, and capital-cost pressure.
- **Rollover impulse:** weakening infrastructure demand, financing conditions, theme leadership, and leverage willingness.

High vulnerability alone can produce `buildout_fragile`; it cannot declare `rollover_confirming` or `break_confirmed`. Those states require an ordered sequence and independent confirmation.

## State policy

| State | Required interpretation |
| --- | --- |
| `normal` | Neither supported composite is elevated. |
| `buildout_fragile` | Vulnerability is high, but rollover is absent or unconfirmed. |
| `prebreak_pressure` | Vulnerability and rollover are elevated, with multiple rollover families but no direct real-economy confirmation. |
| `rollover_confirming` | At least one direct real-economy family and one independent rollover family confirm. |
| `break_confirmed` | Severe rollover, direct real-economy confirmation, and at least two independent rollover confirmations. |
| `ordinary_correction_or_early_rollover` | Rollover-like market pressure without high structural vulnerability. |
| `insufficient_data` | Less than 45% of either composite is supported, or no direct real-economy family is available. Proxy evidence remains visible but cannot resolve the state. |

Broad credit, volatility, breadth, and trend remain the governing confirmation layer for liquidation. This monitor cannot independently create an allocation change.

## Signal families

The current transparent implementation displays eight families, separated by ownership.

Native upstream break evidence:

1. capital intensity / overbuild;
2. financing / reflexivity;
3. monetization / efficiency;
4. infrastructure-demand rollover;

Existing Trade-bot confirmation context:

5. theme-market break;
6. margin leverage;
7. rates / capital competition;
8. equity concentration.

Upstream composites are normalized only across directly observed upstream families.
`vulnerability_coverage` and `rollover_coverage` therefore measure direct evidence
coverage, not the availability of convenient market proxies. Missing direct evidence is
not converted into a neutral zero and is not silently replaced by narrative judgment.

## Point-in-time inputs

The optional fundamentals input is a long-form CSV with at least:

```text
available_date,period_end,entity,metric,value
```

`available_date`, not `period_end`, governs historical eligibility. A filing or observation published after the requested as-of date is excluded even if its economic period ended earlier.

The optional structured-event input contains:

```text
event_date,event_type,entity,direction,severity,confidence,source_quality,cycle_family,decay_half_life_days
```

Events are filtered as of the observation date and decay by half-life. Constructive events offset risk events. Automatic news classification remains auditable, decaying, and zero-authority.

## Explicit missing confirmations

The dashboard always names major upstream gaps, including:

- hyperscaler capex revisions;
- AI supplier orders and backlog;
- GPU or cloud-compute pricing;
- AI-specific credit spreads;
- FINRA margin-debt rollover;
- frontier-lab revenue, margin, and cash burn.

As of V0/V1, market prices, broad credit, and public rates provide useful coverage, but they do not make the missing series observed. The panel should be read as a map of what is known and what would need to change, not as a complete current-cycle diagnosis.

## Outputs

`run-cycle-tracker` persists:

- `capital_cycle_summary.csv`;
- `capital_cycle_families.csv`;
- `capital_cycle_signals.csv`;
- `capital_cycle_events.csv`;
- `capital_cycle_missing_confirmations.csv`;
- `capital_cycle_lead_times.csv`.

All outputs carry zero allocation authority. Lead-time ranges are labeled `conceptual_not_fitted`.

## Promotion boundary

No fitted capital-cycle probability or allocation effect has been promoted. Before any such use, the feature family must be reconstructed across multiple non-AI analogues, including telecom/IT infrastructure, housing/financial leverage, shale/energy, mining/commodities, and speculative growth where data permit. Validation must separately target:

- forward theme drawdowns;
- later economic rollover;
- incremental information beyond existing trend, breadth, credit, volatility, and rates;
- nested walk-forward and sealed outer holdouts;
- leave-episode-out stability;
- ablation, cost, and utility tests;
- prospective shadow monitoring.

The plausible outcome that economic rollover is more predictable than near-term equity timing is acceptable. It would keep the monitor useful for human review without granting it sizing authority.

## Validation harness

Run the prior-only validation separately from the current monitor:

```bash
poetry run trade-bot validate-capital-cycle \
  --fundamentals point_in_time_fundamentals.csv \
  --events governed_events.csv \
  --economic-events economic_rollover_events.csv \
  --episode-manifest analogue_episodes.csv
```

The harness evaluates two market targets—a 10% theme loss within 63 sessions and a 20% loss within 126 sessions—and optional objective economic-rollover events over two and four quarters. It compares:

- the historical event-rate forecast;
- existing Trade-bot market-observable families as a frozen comparator;
- direct upstream vulnerability evidence;
- direct upstream rollover evidence;
- the combined direct upstream family;
- the sequential state logic;
- the frozen existing-market baseline plus direct upstream incremental evidence.

Regularization is selected only within each outer training fold. Test labels must mature before the next fold starts. Observations are weighted so each episode contributes comparable total weight rather than allowing a long crisis window to masquerade as hundreds of independent events.

The validation job writes:

- `capital_cycle_origin_panel.csv`;
- `capital_cycle_oos_predictions.csv`;
- `capital_cycle_diagnostics.csv`;
- `capital_cycle_episode_diagnostics.csv`;
- `capital_cycle_leave_episode_out.csv`;
- `capital_cycle_uncertainty.csv`;
- `capital_cycle_ablation_summary.csv`;
- `capital_cycle_utility_summary.csv`;
- `capital_cycle_readiness.csv`.

The utility table applies at most a 15% hypothetical defensive adjustment to overlapping forward outcomes. It is explicitly a regret/avoided-drawdown diagnostic, not a portfolio backtest. If no historical direct upstream observations exist, the upstream and full-information claim fails readiness closed even when proxy-only market results look favorable.

### Initial proxy-only read and boundary correction: 2026-08-19

The first quarterly-origin run produced 75 origins across seven placeholder calendar
clusters. It could not evaluate an upstream or sequential candidate because no historical
direct upstream family was available and the state therefore remained unresolved.

The available market-only evidence did not establish useful incremental forecasting value:

- for a 10% loss within 63 sessions, the existing-market variant had only a 0.003 median Brier improvement versus the event rate, a negative mean improvement, and a negative episode-bootstrap median;
- vulnerability, rollover, and their unsequenced interaction did not improve the 63-session target;
- the small positive 126-session rollover result was approximately zero on average, had a bootstrap interval spanning zero, and had weak directional discrimination;
- every bounded 15% defensive diagnostic had negative mean incremental forward return;
- approximately 72% of the positive measured lift was concentrated in one placeholder episode cluster.

This was a useful negative control: rearranging existing price, broad-credit,
concentration, and rates evidence did not validate the proposed upstream insight. Those
proxy composites were consequently removed from the candidate set. The current harness
evaluates the existing-market model only as a frozen comparator; `upstream_vulnerability`,
`upstream_rollover`, `upstream_combined`, `sequential_state`, and `full_incremental` fail
closed until direct histories exist. The result remains blocked by absent direct histories,
absent objective economic-rollover labels, placeholder rather than named analogue
episodes, episode concentration, and an immature prospective shadow record.

## Governed historical-data program

The data program is now explicitly staged so a missing upstream read cannot be repaired
by substituting convenient market proxies.

1. `episode_manifest.csv` preregisters three initial breaks—housing/mortgage, mining
   capex, and shale capex—two negative controls—housing recovery and cloud buildout—
   plus dot-com internet speculation and telecom/fiber capacity as two subepisodes in
   one shared statistical cluster. The two dot-com pathways are never counted as two
   independent crises.
2. `source_registry.csv` assigns one primary-source acquisition plan to each of six
   direct families within every episode.
3. `direct_observations.csv` admits a value only when `available_date >= period_end`,
   the source is traceable, and the raw vintage is an archived release, filed fact, or
   immutable transaction record.
4. The feasibility gate requires at least 60% mean direct-family coverage overall,
   at least 50% within every episode, and each family in at least three episodes.
5. Objective economic labels are generated only when preregistered component rules have
   enough distinct releases and the required number of independent components persists.
6. Economic forecasting precedes equity timing. Strategy utility remains locked until
   both historical gates and prospective shadow evidence pass.

Commands:

```bash
poetry run trade-bot fetch-capital-cycle-census-history
poetry run trade-bot fetch-capital-cycle-sec-filings
poetry run trade-bot fetch-capital-cycle-source-documents --episode dotcom_telecom_capacity_1994_2004
poetry run trade-bot import-capital-cycle-fcc-capacity
poetry run trade-bot import-capital-cycle-sec-companyfacts
poetry run trade-bot build-capital-cycle-ai-evidence --as-of YYYY-MM-DD
poetry run trade-bot build-capital-cycle-feasibility
poetry run trade-bot build-capital-cycle-economic-targets
poetry run trade-bot freeze-capital-cycle-trade-bot-baseline
poetry run trade-bot save-capital-cycle-ai-vintage \
  --observations research/capital_cycle/current_ai_observations.csv \
  --as-of YYYY-MM-DD
```

### Feasibility result: 2026-08-20

The first direct reconstruction archived 76 official Census release files and extracted
221 admissible point-in-time housing observations. The official FCC August 2001
*Trends in Telephone Service* report adds 22 telecom-capacity observations from Table
18.3, producing 243 governed historical rows. It covers starts and permits in the housing
demand family, housing completions, and telecom digital-link/working-channel capacity.
Because all 1990-2000 FCC history was published together in August 2001, every extracted
value enters the panel at that release date rather than being backdated. This makes the
series useful for post-break recognition but provides almost no early-warning coverage.

After adding two dot-com subepisodes, mean six-family coverage is 9.7%. Coverage is
present in the housing break, housing-recovery control, and very late in the telecom
capacity subepisode; mining, shale, cloud, and internet-speculation remain at zero. The
feasibility gate therefore fails, and no objective episode label is resolved.

This is a stop at the economic-model boundary, not a conclusion that direct evidence is
useless. The source audit distinguishes three different conditions:

- **reconstructed:** archived Census housing releases;
- **identified but locally blocked or pending:** SEC Company Facts, Baker Hughes rig
  counts, ALFRED vintages, USGS annual publications, and EIA report archives;
- **genuine comparability gap:** standardized cloud utilization and several issuer-level
  commitment-overhang disclosures.

The official SEC Company Facts endpoint returned HTTP 403 from the current execution
environment. Governed local importers therefore accept raw official files without
weakening vintage controls, and a second path parses standardized facts directly from
hash-preserved official inline-XBRL filings. A filing-discovery collector now follows SEC
submissions history files into pre-XBRL 10-K and 10-Q HTML for dot-com reconstruction.
An official-URL seed manifest provides deterministic Global Crossing and Level 3 filing
addresses when the submissions endpoint is unavailable. The current execution environment
also receives HTTP 403 for those SEC Archive URLs, so the URLs and failed acquisition
inventory are retained but their contents are not claimed as collected evidence.
This access limitation must not be misreported as evidence absence.
The frozen Trade-bot comparator contains 291 latest-in-day saved market dates from
2007-07-11 through 2026-08-19. Its manifest states that reconstructed historical snapshots
are not contemporaneous model-version history.

The original August 2026 prospective AI vintage remains immutable at zero of six because
it was initialized before the collection pipeline existed. That record is not overwritten.
The 2026-08-20 evidence build now contains 123 point-in-time rows. Five families—demand,
capacity, financing, monetization, and commitment overhang—pass the explicit freshness,
row-count, and entity-breadth rules. Pricing/utilization remains unavailable: the public
record contains scattered operational-status statements but not a comparable, fresh,
multi-entity series of realized compute price and productive utilization. Calling that
family available would manufacture evidence.

The evidence build reuses primary-source ledgers and hash-preserved filings from the AI
financial-system monograph. It distinguishes observed from usable: a family needs at least
two eligible rows, two entities, and family-specific freshness. `save-capital-cycle-ai-vintage`
now rejects an empty input unless `--allow-empty` is explicitly supplied, preventing the
earlier initialization mistake from recurring silently. All acquisition and coverage
artifacts retain zero allocation authority.
