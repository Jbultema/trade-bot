# Parallel Research Staging Protocol

This directory isolates concurrent research from the canonical monograph database and
publication artifacts.

## Write authority

Research lanes may write only inside their assigned lane directory. They may not edit:

- `research/ai_financial_system/data/`;
- `research/ai_financial_system/schema.sql`;
- canonical chapter HTML;
- database builders or shared validators;
- `reports/ai_financial_system/`;
- final DOCX files or delivery manifests.

The primary lane is the only canonical integrator.

## Provisional identity

Every staged row uses a lane-prefixed provisional ID. A provisional ID has no canonical
meaning and may be merged, split or rejected during integration. Existing canonical
IDs may be referenced but never altered.

## Required row provenance

Every staged factual claim includes:

- source ID or proposed source ID;
- URL or archived path;
- document date;
- exact locator;
- evidence grade;
- transaction or measurement state;
- as-of date; and
- known limitation.

Rows without locators are research leads, not proposed claims.

## Integration sequence

1. Verify the source and hash or archive it.
2. Resolve entity, project, contract and exposure identities.
3. Test whether the observation duplicates a canonical component.
4. Reconcile amount, currency, time and recognition state.
5. Test the proposed evidence grade and confidence.
6. Register or reject the source.
7. Assign canonical IDs.
8. Load into a temporary database.
9. Run foreign-key, source-grade, overlap and chapter-gate validation.
10. Rebuild the canonical database only after the temporary integration passes.

Before substantive review, the primary lane runs:

```bash
python3 research/ai_financial_system/scripts/validate_staging_packet.py \
  research/ai_financial_system/staging/<lane>
```

This gate verifies the sealed manifest, file hashes, row and line counts, CSV shape,
provisional-key uniqueness, canonical source URL matching, and staged claim
references. A packet that is still being edited or whose manifest is stale must fail.
Passing this structural gate does not establish that the proposed interpretation is
correct; reciprocal review and primary-lane adjudication remain mandatory.

## Acceptance states

- `accepted` — canonicalized without substantive change;
- `accepted_with_revision` — evidence retained but claim, amount or state changed;
- `duplicate_component` — linked to an existing exposure and marked nonadditive;
- `research_lead` — useful but not sufficient for a claim;
- `blocked` — material evidence remains unavailable;
- `rejected` — provenance, logic or scope does not support inclusion.

Staged work is never cited as a delivered monograph artifact until canonical
integration and publication validation pass.
