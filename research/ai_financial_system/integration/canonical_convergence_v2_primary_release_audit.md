# Canonical convergence v2 primary release audit

**Status:** `CHANGES_REQUESTED_RELEASE_SHAPE_AND_SEMANTIC_NORMALIZATION`

**Scope:** primary release-boundary review of the sealed staging candidate at
`research/ai_financial_system/staging/canonical_convergence_v2/`.

This audit does not reverse the candidate's structural accomplishments. It
determines whether those accomplishments prove that the candidate can replace
the canonical research system described by
`canonical_convergence_v2_task.md` and
`second_half_canonical_integration_design.md`.

## Evidence that passes

- All 11 accepted author/reviewer packet pairs remain hash-bound.
- The input registry passes its fail-closed prebuild gate.
- The packet preserves 894 accepted artifacts and dispositions all 1,521,360
  accepted CSV scalar fields.
- The candidate SQLite database returns `integrity_check=ok` and no foreign-key
  violations.
- The dedicated convergence validator passes 14 of 14 controls.
- The shared staging validator passes literally with zero errors across 999
  manifested files and 121,070 CSV rows.
- Two clean builds are byte-identical:
  - database SHA-256:
    `fe0d9216ee7f2d290bb173f723f980db83c670cbc3543d23d187d84d0445b292`;
  - candidate-tree SHA-256:
    `d697953bc804a76980c1a044d88d0232bba39eabfd2c6c907b7827465f306ae1`.
- The 60-file protected canonical baseline remains unchanged.

These facts establish a deterministic, lossless staging archive and queryable
review surface. They do not establish release readiness.

## Blocking release findings

### 1. The candidate is not a release-shaped canonical data tree

The candidate places the existing flat canonical CSVs under
`candidate/candidate_data/canonical_baseline/`, accepted packet artifacts under
`candidate/candidate_data/accepted_inputs/`, review manifests under
`candidate/candidate_data/review_bindings/`, and only the generated
`research.db` at the candidate-data root.

The current canonical builder expects the governed flat CSVs under
`research/ai_financial_system/data/`. Replacing that directory with the
candidate tree would therefore break the current build contract. Copying only
the candidate database would create a database that cannot be regenerated from
the released canonical data tree.

### 2. `candidate_schema.sql` is an extension fragment, not a full canonical schema

The candidate database begins as a byte copy of the existing canonical
database. The builder then executes the convergence extension schema and
dynamically generated `accepted__*` table/view statements. The emitted
`candidate_schema.sql` contains those extensions, not the complete original
canonical schema plus extensions and loader contract.

It therefore cannot independently recreate the released database from an empty
SQLite file and a governed released data tree.

### 3. The 68 semantic endpoints are raw pass-through views

Every row in `convergence_semantic_endpoints` joins to a source table whose
`normalized_relationship_state` is:

`raw_transport_preserved;multi_value_fields_normalized_separately`

Representative endpoint definitions are of the form:

```sql
CREATE VIEW ch16_counterparty_roles AS
SELECT * FROM accepted__chapter_16__counterparty_role_matrix;
```

The dynamic source tables use `__source_row_ordinal` as their only primary key
and load every source column as `TEXT`. The views do not add stable semantic
primary keys, canonical foreign keys, typed state constraints, relationship
tables, or publication-eligibility constraints. Queryability is useful, but a
renamed transport table is not the normalized domain required by the governing
design.

### 4. Field disposition is syntactic, not substantive canonical integration

The field-disposition population is:

| Disposition | Fields |
|---|---:|
| `canonical_destination` | 1,349,874 |
| `governed_hold` | 105,252 |
| `typed_junction_destination` | 66,234 |

The builder derives these states from filename, column-name, blank-value and
keyword heuristics. Most `canonical_destination` values point back to the
generated raw `accepted__*` table, including audit and disposition artifacts.
An explicit blank is automatically materialized as an authority hold.

This is adequate as a loss-accounting transport audit. It is not evidence that
1,349,874 fields received a substantively accepted canonical economic
destination.

### 5. The 12,060 “recomputations” are accepted-value identity replays

All 12,060 rows in `derived_value_recomputation.csv` use:

`accepted_value_identity_replay_hash_bound_to_fresh_review`

The accepted value and “recomputed” value are copied from the same staged
field. This proves preservation and binds the accepted review lineage. It does
not independently recompute the formula, denominator, population, units,
tolerance, or result. The build report appropriately narrows this claim, but
the governing convergence requirement calls for recomputation of accepted
derived values.

### 6. The canonical economic populations remain unchanged

The candidate retains the baseline populations of 156 sources, 427 claims, 581
evidence links, 252 unknowns, 36 economic exposures and 47 chapter handoffs.
That is appropriate where new packets are deliberately noncanonical, but it
also confirms that accepted second-half domains have not been integrated into
a stable canonical domain layer. They remain packet-scoped tables and views.

## Release decision

Do not replace the canonical CSVs, schema, database or manifests with this
candidate.

Retain v2 as a deterministic lossless staging archive and as evidence for the
next convergence build. Its manifest and validation results must not be
described as a canonical release acceptance.

## Required v3 convergence

1. Produce a release-shaped candidate containing:
   - the complete flat canonical data population;
   - governed accepted-input and review-binding subtrees;
   - normalized domain exports;
   - a complete schema;
   - a clean-build loader;
   - the rebuilt database;
   - database, data-tree, schema and release manifests.
2. Rebuild from an empty database using only candidate-release files. Do not
   seed by copying the prior SQLite database.
3. Preserve existing canonical tables and implement the accepted new domains
   as stable tables with documented semantic primary keys, scalar foreign keys,
   checks and explicit junctions.
4. Make the analyst-facing endpoints project from those normalized canonical
   tables, not directly from packet transport mirrors.
5. Separate byte-preservation disposition from substantive canonical
   disposition. Require explicit domain mapping for claims of canonical
   projection.
6. Replace identity replay with formula-level recomputation where a derived
   metric is asserted as recomputed. Where formula reconstruction is not
   available, retain an explicit reviewed-value replay state rather than a
   recomputation pass.
7. Reconcile every new normalized row to its packet provenance, source and
   applicable claim or governed method boundary.
8. Demonstrate:
   - empty-root deterministic twins;
   - database integrity and foreign keys;
   - exact source-to-release field coverage;
   - canonical identity and duplicate-economic-exposure controls;
   - full rebuild parity;
   - protected no-write before release;
   - an atomic release plan with rollback evidence.
9. Obtain a fresh nonauthor review of the v3 release candidate before any
   canonical mutation or publication dependency is advanced.

## Publication boundary

Final chapter prose, DOCX files, the executive summary, the essay and the three
PDF packages must remain bound to the existing accepted canonical state until a
release-shaped convergence candidate passes the requirements above. Staging v2
may be consulted as reviewed evidence, but its endpoint existence does not by
itself authorize publication assertions.
