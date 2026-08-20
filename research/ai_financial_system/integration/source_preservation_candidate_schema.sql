-- Candidate-only source preservation schema.
-- This file does not authorize or mutate the canonical database.

PRAGMA foreign_keys = ON;

CREATE TABLE source_versions (
    source_version_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    original_url TEXT,
    resolved_url TEXT,
    retrieval_timestamp_utc TEXT NOT NULL,
    publication_date TEXT,
    filing_date TEXT,
    effective_date TEXT,
    mime_type TEXT NOT NULL,
    byte_count INTEGER CHECK (byte_count IS NULL OR byte_count >= 0),
    preservation_class TEXT NOT NULL CHECK (
        preservation_class IN (
            'P1_public_primary_document',
            'P2_mutable_official_page_or_api',
            'P3_licensed_or_copyrighted_secondary',
            'P4_internal_reconstruction_or_derived_dataset'
        )
    ),
    preservation_method TEXT NOT NULL,
    content_sha256 TEXT,
    archived_path TEXT,
    immutable_external_archive_id TEXT,
    http_status TEXT,
    response_metadata_state TEXT NOT NULL,
    access_or_licensing_restriction TEXT NOT NULL,
    preservation_status TEXT NOT NULL CHECK (
        preservation_status IN (
            'preserved_complete',
            'preserved_response_plus_metadata',
            'preserved_derived_with_original',
            'metadata_only_rights_boundary',
            'immutable_external_archive_verified',
            'retrieval_failed',
            'access_restricted',
            'content_changed_requires_version_review',
            'not_yet_preserved',
            'not_applicable_internal_derivation'
        )
    ),
    exception_reason TEXT NOT NULL DEFAULT '',
    supersedes_source_version_id TEXT REFERENCES source_versions(source_version_id),
    notes TEXT NOT NULL DEFAULT '',
    CHECK (
        (content_sha256 IS NOT NULL AND archived_path IS NOT NULL)
        OR preservation_status IN (
            'metadata_only_rights_boundary',
            'immutable_external_archive_verified',
            'retrieval_failed',
            'access_restricted',
            'not_yet_preserved',
            'not_applicable_internal_derivation'
        )
    )
);

CREATE TABLE source_derivatives (
    source_derivative_id TEXT PRIMARY KEY,
    source_version_id TEXT NOT NULL REFERENCES source_versions(source_version_id),
    derivative_type TEXT NOT NULL,
    derivative_path TEXT NOT NULL,
    derivative_content_sha256 TEXT NOT NULL,
    derivative_byte_count INTEGER NOT NULL CHECK (derivative_byte_count >= 0),
    extractor_name TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    extractor_parameters TEXT NOT NULL,
    validation_status TEXT NOT NULL CHECK (
        validation_status IN ('pass', 'fail', 'review_required')
    ),
    evidentiary_boundary TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE claim_locator_reviews (
    locator_review_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    cited_locator TEXT NOT NULL,
    source_version_id TEXT NOT NULL REFERENCES source_versions(source_version_id),
    resolved_locator TEXT NOT NULL,
    principal_claim_role TEXT NOT NULL CHECK (
        principal_claim_role IN ('principal', 'supporting', 'context', 'method')
    ),
    support_type TEXT NOT NULL CHECK (
        support_type IN (
            'direct',
            'corroborating',
            'contradicting',
            'context',
            'method'
        )
    ),
    semantic_result TEXT NOT NULL CHECK (
        semantic_result IN (
            'supported',
            'partially_supported',
            'unsupported',
            'ambiguous',
            'locator_not_found',
            'review_pending'
        )
    ),
    unsupported_proposition_or_required_narrowing TEXT NOT NULL DEFAULT '',
    reviewer TEXT NOT NULL,
    review_date TEXT NOT NULL,
    review_artifact TEXT NOT NULL,
    publication_eligibility TEXT NOT NULL CHECK (
        publication_eligibility IN (
            'eligible',
            'eligible_with_explicit_qualification',
            'blocked',
            'pending'
        )
    ),
    supersedes_locator_review_id TEXT REFERENCES claim_locator_reviews(locator_review_id),
    notes TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (claim_id, source_id, cited_locator)
        REFERENCES evidence_links(claim_id, source_id, locator)
);

CREATE INDEX source_versions_source_idx
    ON source_versions(source_id, retrieval_timestamp_utc);
CREATE INDEX claim_locator_reviews_claim_idx
    ON claim_locator_reviews(claim_id, publication_eligibility);
