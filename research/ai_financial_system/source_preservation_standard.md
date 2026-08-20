# Source preservation and reproducibility standard

## Purpose

A URL and retrieval date identify where evidence was found; they do not preserve
what the source contained when a claim was made. Publication reliance therefore
requires a lawful, content-addressed preservation record or an explicit preservation
exception.

This standard governs source preservation separately from claim monitoring. A frozen
filing does not become stale merely because time passes, while the current state of a
company, project, or contract may require a later filing and a new claim version.

## Required preservation record

Each relied-on external source receives:

- canonical `source_id`;
- original URL and final resolved URL;
- retrieval timestamp in UTC;
- publisher and document title;
- publication/filing/effective date where available;
- MIME type and byte length;
- preservation method;
- SHA-256 content hash over the preserved bytes;
- local archive path or lawful external immutable-archive identifier;
- HTTP status and relevant response metadata where available;
- access or licensing restriction;
- extraction/OCR method and version, if used;
- relationship to the cited locator;
- preservation status and exception reason.

The hash must cover the actual frozen artifact, not a URL string, citation text, or
download timestamp.

## Preservation classes

### Class P1 — public primary document

Examples: SEC filing, official court or regulatory filing, bond document, government
docket, official company PDF, tariff, prospectus, earnings release.

Preserve the complete retrieved document when lawful and technically possible.
Retain the original format. Any normalized text or OCR is a derived artifact with its
own hash and a pointer to the original.

### Class P2 — mutable official page or API response

Preserve the response bytes plus request parameters and response metadata. If the
page depends on client-side rendering, preserve both the raw response and a rendered
derivative where lawful. Never treat a screenshot alone as a machine-auditable source.

### Class P3 — licensed or copyrighted secondary reporting

Do not create an unauthorized local republication. Preserve citation metadata,
permitted excerpts, access date, locator, and—where available—a lawful archive or
licensed-document identifier. Record `full_content_not_archived_rights_boundary`.
The claim registry must not imply that a content hash exists when only citation
metadata was retained.

### Class P4 — internal reconstruction or derived dataset

Preserve the canonical input IDs, transformation code/version, parameters, execution
environment, output bytes, and hashes. An `internal://` path is method provenance,
not primary evidence. Every principal factual input must still resolve to external
evidence or an explicit unknown.

## Status vocabulary

- `preserved_complete`
- `preserved_response_plus_metadata`
- `preserved_derived_with_original`
- `metadata_only_rights_boundary`
- `immutable_external_archive_verified`
- `retrieval_failed`
- `access_restricted`
- `content_changed_requires_version_review`
- `not_yet_preserved`
- `not_applicable_internal_derivation`

Blank does not mean preserved, and an accessible URL does not mean archived.

## Integrity and versioning controls

1. Preservation is append-only. A changed artifact receives a new version row and
   hash; it never overwrites the prior record.
2. Redirects, HTML wrappers, and downloaded attachments are distinct artifacts where
   each contributes evidentiary content.
3. The cited locator is checked against the preserved artifact.
4. Duplicate hashes may share one blob but retain separate source/version metadata.
5. A source supersession does not automatically supersede every claim it supports.
6. Effective dates, publication dates, retrieval dates, and version-observed dates
   remain distinct.
7. Hash success establishes byte identity, not truth, completeness, evidence grade,
   or independent corroboration.
8. Resume or cache reuse must match both the canonical source identity and the exact
   original URL. A source identity may lawfully bind several constituent artifacts
   (for example, a filing wrapper, exhibit, presentation, and financial-statement
   attachment); retrieval of one constituent must never be treated as preservation
   of another merely because they share a `source_id`.

## Publication gates

Before final publication:

- every Grade A principal claim must resolve to a P1/P2 preserved artifact or a
  documented, reviewer-accepted preservation exception;
- every Grade B principal claim must resolve to a lawful P3 preservation record;
- every Grade C reconstruction must resolve to preserved inputs and a reproducible
  P4 transformation;
- the source register must report preservation coverage by evidence grade and
  principal/nonprincipal claim role;
- all missing principal-source preservation must appear in the publication exception
  register with owner, reason, attempted route, consequence, and next action.

The gate is risk-weighted rather than a misleading raw percentage: one missing
principal bond indenture can be more consequential than dozens of preserved context
pages.

## Operational boundary

Automated collection must respect access controls, rate limits, publisher terms, and
copyright. The preservation pipeline must fail closed: a failed or prohibited
download remains an explicit exception and may not be converted into a synthetic
hash, cached snippet, or implied archive.
