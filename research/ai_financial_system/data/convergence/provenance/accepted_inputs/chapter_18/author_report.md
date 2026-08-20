# Chapter 18 immutable source-metadata rebase author report

**Status date:** 2026-07-26  
**Author verdict:** `READY_FOR_FRESH_UNINVOLVED_METADATA_ACCEPTANCE_WITH_RETAINED_HOLDS`  
**Author validation is acceptance:** `NO`  
**Canonical integration:** `NOT_AUTHORIZED`  
**Chapter/DOCX/publication authority:** `NOT_AUTHORIZED`  
**Coefficient, predictive, causal, comparative, and model use:** `BLOCK`

## Governing bindings

All four accepted manifests and the frozen canonical database matched their
task-pinned hashes before and after the sequential build:

- `intrinsic_author_manifest`: `d11f8bd03062b2abfbcb85d3c351ea614eb2d06a3704ed15b6e571c2e758b8de` (`PASS`)
- `intrinsic_review_manifest`: `d4e2646953001f69c17aeddeee7e80af26d4260c9389b8e6fd7f3b0c30159cf8` (`PASS`)
- `corrected_preservation_author_manifest`: `3fad6122a81e115a352e20bfdb3086be1a25d5ea6547202c97d6ac2020b11867` (`PASS`)
- `corrected_preservation_review_manifest`: `d4e1d026f7743032e77f98a6ed1424ccd4270bb5a5188a3ed1a5d98c44eb8dd9` (`PASS`)
- `frozen_canonical_database`: `38b990b2228dc62ebfa20a01ef5855694fad7fef47156535def138fa02ed48a6` (`PASS`)

The sealed candidate database is byte-identical to the frozen canonical
database at `38b990b2228dc62ebfa20a01ef5855694fad7fef47156535def138fa02ed48a6` (1777664 bytes), and both
read-only SQLite integrity checks return `ok`. This packet performs no
canonical mutation.

## Immutable intrinsic population

All 50 manifest-declared intrinsic author artifacts are copied byte-for-byte
under `intrinsic_artifacts/`. The accepted packet remains exactly:

- 128 analytical records;
- 214 record/claim junction links;
- 20 arithmetic and
  nonadditivity controls;
- all 17 earlier intrinsic repairs and both rebuilt-v5 residual repairs.

The lossless disposition covers 1,654 CSV rows and
17,112 scalar fields, each by direct retained identity.
No accepted intrinsic claim, arithmetic result, prose boundary, evidence
grade, support direction, or prohibition is rewritten.

## Accepted source preservation and semantics

The rebase reproduces exactly:

- 58 semantic-locator relationships:
  32 `supported` and
  26 `partially_supported`;
- 18 accepted semantic-scope patches and
  73 ordered claim/source/version/locator
  edges;
- 23 cited source identities and
  25 append-only retrieval attempts;
- 23 preserved artifacts,
  2 original UNCTAD HTTP-403 failures, and
  the distinct successful v0003 alternate;
- 23 derivatives with exact derivative and parent
  hashes.

Every one of 718 accepted preservation-metadata CSV rows and
8,096 scalar fields has exactly one lossless disposition:

- `direct_retained_identity`: 2546
- `governed_hold_destination`: 2838
- `normalized_junction_destination`: 2712

There are no missing fields, populated-to-null changes, many-to-one
overwrites, or authorized rejections. Stable semantic keys and all foreign
keys pass.

## Authority boundary and retained holds

Preservation success, semantic support, evidence grade, claim scope,
analytical eligibility, and publication authority remain separate machine
fields. `partially_supported` is never promoted. Preserved bytes are explicitly
not semantic proof.

All 18 coefficient-transfer blocks and
16 predictive/conceptual-hypothesis blocks
remain in force. Causal transfer, comparative-system use, model use, canonical
integration, DOCX mutation, and publication remain blocked or not authorized
on every authority row. The two failed retrieval attempts, all source
exceptions, unresolved queue rows, and claim-scope limitations remain
append-only governed holds.

## Reproducibility and no-write evidence

Two sequential clean builds compare 94 generated files
and are byte-identical at tree hash `05dd2bdff3f5f0a1aad63a896a09ca7f6ae47b173276bb5120540041423961f3`. The
author lane matches the deterministic core; `validation_results.json` has the
expected post-seal-only variance that replaces the pre-seal deferred-manifest
check with the passing 102-entry sealed-manifest audit. Dedicated validation,
generic staging validation, and Ruff pass.

The protected-input audit covers 327 files across the four
governing packets, canonical database and registries, chapter sources,
document/publication surfaces, and named shared instructions/trackers. Every
protected file is byte-identical before and after the build.

## Recommendation

This packet is ready for a fresh uninvolved metadata acceptance review. It does
not perform primary canonical integration, rewrite Chapter 18, replace a DOCX,
clear coefficient/predictive/comparative holds, or authorize publication.
