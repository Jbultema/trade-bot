-- Append-only source-preservation union v2 candidate; not canonical.
PRAGMA foreign_keys=ON;
CREATE TABLE source_versions (
  source_version_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
  provenance_population TEXT NOT NULL, original_url TEXT,
  final_resolved_url TEXT, retrieval_timestamp_utc TEXT,
  publication_date TEXT, preservation_class TEXT NOT NULL,
  preservation_status TEXT NOT NULL, mime_type TEXT, byte_count INTEGER,
  content_sha256 TEXT, local_archive_path TEXT, preservation_method TEXT,
  response_metadata_state TEXT, access_licensing_boundary TEXT,
  artifact_role TEXT, cited_locator_relationship TEXT,
  exception_reason TEXT, candidate_integration_status TEXT NOT NULL
);
CREATE TABLE source_derivatives (
  derivative_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
  parent_source_version_id TEXT NOT NULL REFERENCES source_versions,
  derivation_method_version TEXT NOT NULL, normalization_rule TEXT,
  derivative_sha256 TEXT NOT NULL, derivative_byte_count INTEGER NOT NULL,
  derivative_path TEXT NOT NULL, locator_aid_status TEXT NOT NULL,
  governing_boundary TEXT NOT NULL, candidate_integration_status TEXT NOT NULL
);
CREATE TABLE source_exceptions (
  exception_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
  source_version_id TEXT NOT NULL REFERENCES source_versions,
  exception_type TEXT NOT NULL, exception_reason TEXT NOT NULL,
  content_sha256 TEXT, local_archive_path TEXT,
  resolution_state TEXT NOT NULL, publication_effect TEXT NOT NULL
);
CREATE TABLE duplicate_blob_reconciliation (
  source_id TEXT PRIMARY KEY,
  private_source_version_id TEXT NOT NULL REFERENCES source_versions,
  chapters_06_09_source_version_id TEXT NOT NULL REFERENCES source_versions,
  private_sha256 TEXT NOT NULL, chapters_06_09_sha256 TEXT NOT NULL,
  private_byte_count INTEGER NOT NULL, chapters_06_09_byte_count INTEGER NOT NULL,
  reconciliation_result TEXT NOT NULL, version_policy TEXT NOT NULL
);
CREATE TABLE carried_private_locator_reviews (
  claim_locator_review_id TEXT PRIMARY KEY, claim_id TEXT NOT NULL,
  evidence_grade TEXT NOT NULL, source_ids TEXT NOT NULL, locator TEXT NOT NULL,
  archived_support TEXT NOT NULL, semantic_verdict TEXT NOT NULL,
  remaining_boundary TEXT NOT NULL, review_provenance TEXT NOT NULL,
  candidate_status TEXT NOT NULL
);
CREATE TABLE claim_locator_states (
  claim_locator_state_id TEXT PRIMARY KEY, claim_id TEXT NOT NULL,
  chapter_id TEXT NOT NULL, source_id TEXT NOT NULL, locator TEXT NOT NULL,
  support_type TEXT NOT NULL, semantic_review_state TEXT NOT NULL,
  semantic_verdict TEXT, remaining_boundary TEXT NOT NULL,
  review_provenance TEXT NOT NULL
);
CREATE TABLE chapter_support_state_projection (
  projection_row_id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL,
  chapter_id TEXT NOT NULL, claim_id TEXT NOT NULL, source_id TEXT NOT NULL,
  locator TEXT NOT NULL, support_type TEXT NOT NULL, support_origin TEXT NOT NULL,
  source_version_id TEXT REFERENCES source_versions,
  source_state_disposition TEXT NOT NULL, semantic_review_authority TEXT NOT NULL,
  candidate_integration_status TEXT NOT NULL
);
CREATE TABLE source_version_delta_audit (
  delta_audit_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
  historical_source_version_id TEXT NOT NULL REFERENCES source_versions,
  appended_source_version_id TEXT NOT NULL REFERENCES source_versions,
  historical_status TEXT NOT NULL, appended_status TEXT NOT NULL,
  alternate_venue_relationship TEXT NOT NULL, current_projection_version_id TEXT
    NOT NULL REFERENCES source_versions,
  semantic_locator_state TEXT NOT NULL, quarantine_excluded INTEGER NOT NULL,
  result TEXT NOT NULL
);
