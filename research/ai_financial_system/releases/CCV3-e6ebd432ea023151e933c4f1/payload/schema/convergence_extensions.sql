PRAGMA foreign_keys = ON;

CREATE TABLE private_investment_accounting_methods (
    method_id TEXT PRIMARY KEY,
    accounting_method TEXT NOT NULL UNIQUE,
    governing_basis TEXT NOT NULL,
    recognition_trigger TEXT NOT NULL,
    income_statement_or_oci_location TEXT NOT NULL,
    cash_at_recognition_default TEXT NOT NULL,
    carrying_value_effect TEXT NOT NULL,
    required_separation TEXT NOT NULL,
    example TEXT NOT NULL,
    provenance_class TEXT NOT NULL,
    notes TEXT NOT NULL
);

CREATE TABLE private_investment_earnings (
    private_investment_earnings_id TEXT PRIMARY KEY,
    investor_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    reporting_period_label TEXT NOT NULL,
    reporting_period_start TEXT,
    reporting_period_end TEXT,
    reporting_period_date_status TEXT NOT NULL,
    investee_entity_id TEXT REFERENCES entities(entity_id),
    investee_scope_description TEXT NOT NULL,
    instrument_scope TEXT NOT NULL,
    accounting_method_id TEXT REFERENCES private_investment_accounting_methods(method_id),
    accounting_method_scope TEXT NOT NULL,
    recognition_location TEXT NOT NULL,
    reported_line_item TEXT NOT NULL,
    pretax_gain_loss_amount REAL,
    currency TEXT NOT NULL,
    unit_scale TEXT NOT NULL,
    amount_precision TEXT NOT NULL,
    amount_unparsed_suffix TEXT NOT NULL,
    recognition_state TEXT NOT NULL,
    cash_received_amount REAL,
    cash_received_state TEXT NOT NULL,
    cash_amount_precision TEXT NOT NULL,
    reported_pretax_income_amount REAL,
    reported_pretax_income_precision TEXT NOT NULL,
    gain_share_of_pretax_income_pct REAL,
    ratio_precision TEXT NOT NULL,
    investee_attribution_state TEXT NOT NULL,
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    source_locator TEXT NOT NULL,
    source_claim_id TEXT REFERENCES claims(claim_id),
    nonadditive_overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0,1)),
    analyst_derived INTEGER NOT NULL CHECK (analyst_derived IN (0,1)),
    integration_status TEXT NOT NULL CHECK (
        integration_status =
            'accepted_bounded_preservation_cleared_other_holds_retained'
    ),
    notes TEXT NOT NULL
);

CREATE TABLE private_investment_earnings_components (
    parent_earnings_id TEXT NOT NULL REFERENCES private_investment_earnings(private_investment_earnings_id),
    component_earnings_id TEXT NOT NULL REFERENCES private_investment_earnings(private_investment_earnings_id),
    component_order INTEGER NOT NULL,
    relationship TEXT NOT NULL,
    additive_within_parent INTEGER NOT NULL CHECK (additive_within_parent IN (0,1)),
    notes TEXT NOT NULL,
    PRIMARY KEY (parent_earnings_id, component_earnings_id)
);

CREATE TABLE private_investment_carrying_value_rollforwards (
    rollforward_id TEXT PRIMARY KEY,
    investor_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    reporting_period_label TEXT NOT NULL,
    reporting_period_start TEXT,
    reporting_period_end TEXT,
    reporting_period_date_status TEXT NOT NULL,
    investee_entity_id TEXT REFERENCES entities(entity_id),
    investee_scope_description TEXT NOT NULL,
    instrument_scope TEXT NOT NULL,
    beginning_carrying_value_amount REAL,
    ending_carrying_value_amount REAL,
    currency TEXT NOT NULL,
    unit_scale TEXT NOT NULL,
    beginning_amount_precision TEXT NOT NULL,
    ending_amount_precision TEXT NOT NULL,
    reconciliation_state TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    source_locator TEXT NOT NULL,
    integration_status TEXT NOT NULL CHECK (
        integration_status =
            'accepted_bounded_preservation_cleared_other_holds_retained'
    ),
    notes TEXT NOT NULL
);

CREATE TABLE private_investment_rollforward_components (
    rollforward_component_id TEXT PRIMARY KEY,
    rollforward_id TEXT NOT NULL REFERENCES private_investment_carrying_value_rollforwards(rollforward_id),
    component_type TEXT NOT NULL,
    amount REAL,
    currency TEXT NOT NULL,
    unit_scale TEXT NOT NULL,
    amount_precision TEXT NOT NULL,
    amount_unparsed_suffix TEXT NOT NULL,
    cash_state TEXT NOT NULL,
    income_aoci_balance_sheet_class TEXT NOT NULL,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0,1)),
    overlap_group TEXT,
    notes TEXT NOT NULL
);

CREATE TABLE private_investment_earnings_bridges (
    bridge_id TEXT PRIMARY KEY,
    investor_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    reporting_period_label TEXT NOT NULL,
    reporting_period_start TEXT,
    reporting_period_end TEXT,
    reporting_period_date_status TEXT NOT NULL,
    reported_operating_income_amount REAL,
    reported_pretax_income_amount REAL,
    identified_private_investment_gain_amount REAL,
    arithmetic_pretax_residual_amount REAL,
    private_gain_share_pretax_pct REAL,
    cash_flow_reversal_amount REAL,
    cash_flow_reversal_precision TEXT NOT NULL,
    cash_flow_reversal_perimeter TEXT NOT NULL,
    currency TEXT NOT NULL,
    unit_scale TEXT NOT NULL,
    selected_gain_precision TEXT NOT NULL,
    residual_precision TEXT NOT NULL,
    ratio_precision TEXT NOT NULL,
    tax_effect_state TEXT NOT NULL,
    analyst_derived_state TEXT NOT NULL,
    integration_status TEXT NOT NULL CHECK (
        integration_status =
            'accepted_bounded_preservation_cleared_other_holds_retained'
    ),
    notes TEXT NOT NULL
);

CREATE TABLE private_investment_bridge_components (
    bridge_id TEXT NOT NULL REFERENCES private_investment_earnings_bridges(bridge_id),
    private_investment_earnings_id TEXT NOT NULL REFERENCES private_investment_earnings(private_investment_earnings_id),
    component_role TEXT NOT NULL,
    PRIMARY KEY (bridge_id, private_investment_earnings_id)
);

CREATE TABLE private_investment_population_observations (
    population_observation_id TEXT PRIMARY KEY,
    investor_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    issuer_label TEXT NOT NULL,
    reporting_period_label TEXT NOT NULL,
    reporting_period_start TEXT,
    reporting_period_end TEXT,
    reporting_period_date_status TEXT NOT NULL,
    investee_entity_id TEXT REFERENCES entities(entity_id),
    investee_scope_description TEXT NOT NULL,
    accounting_method_id TEXT REFERENCES private_investment_accounting_methods(method_id),
    accounting_method_scope TEXT NOT NULL,
    observation_class TEXT NOT NULL,
    material_private_gain_state TEXT NOT NULL,
    operating_nonoperating_state TEXT NOT NULL,
    carrying_value_amount REAL,
    cash_proceeds_amount REAL,
    cumulative_upward_adjustments_amount REAL,
    current_period_private_gain_loss_amount REAL,
    gain_share_of_pretax_pct REAL,
    named_position_cost_amount REAL,
    reported_pretax_income_amount REAL,
    currency TEXT NOT NULL,
    unit_scale TEXT NOT NULL,
    amount_precision_state TEXT NOT NULL,
    investee_attribution_state TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    source_url TEXT NOT NULL,
    integration_status TEXT NOT NULL CHECK (
        integration_status =
            'accepted_bounded_preservation_cleared_other_holds_retained'
    ),
    notes TEXT NOT NULL
);

PRAGMA foreign_keys = ON;

CREATE TABLE private_source_archive_versions (
    archive_record_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    source_url TEXT NOT NULL,
    resolved_url TEXT NOT NULL,
    retrieval_timestamp_utc TEXT NOT NULL,
    archived_path TEXT NOT NULL,
    content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
    byte_count INTEGER NOT NULL CHECK (byte_count >= 0),
    mime_type TEXT NOT NULL,
    preservation_class TEXT NOT NULL CHECK (
        preservation_class = 'P1_public_primary_document'
    ),
    preservation_method TEXT NOT NULL,
    http_status TEXT NOT NULL,
    response_metadata_state TEXT NOT NULL CHECK (
        response_metadata_state =
            'body_preserved_headers_not_separately_frozen'
    ),
    governing_evidence_state TEXT NOT NULL CHECK (
        governing_evidence_state = 'original_inline_xbrl_governs'
    ),
    locator_review_state TEXT NOT NULL CHECK (
        locator_review_state =
            'carried_forward_from_sealed_nonauthor_claim_review'
    ),
    integration_status TEXT NOT NULL CHECK (
        integration_status = 'accepted_bounded_preservation_rebase'
    ),
    UNIQUE (source_id, content_sha256)
);

CREATE TABLE private_source_text_derivatives (
    derivative_record_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    archive_record_id TEXT NOT NULL
        REFERENCES private_source_archive_versions(archive_record_id),
    original_archived_path TEXT NOT NULL,
    original_content_sha256 TEXT NOT NULL
        CHECK (length(original_content_sha256) = 64),
    derived_text_path TEXT NOT NULL,
    derived_text_sha256 TEXT NOT NULL
        CHECK (length(derived_text_sha256) = 64),
    derived_byte_count INTEGER NOT NULL CHECK (derived_byte_count >= 0),
    extraction_method TEXT NOT NULL,
    normalization_rule TEXT NOT NULL,
    preservation_status TEXT NOT NULL CHECK (
        preservation_status = 'preserved_derived_with_original'
    ),
    evidentiary_boundary TEXT NOT NULL CHECK (
        evidentiary_boundary =
            'locator_aid_only_original_inline_xbrl_governs'
    ),
    integration_status TEXT NOT NULL CHECK (
        integration_status = 'accepted_bounded_preservation_rebase'
    ),
    UNIQUE (source_id, derived_text_sha256)
);

CREATE TABLE private_claim_locator_reviews (
    claim_locator_review_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    locator TEXT NOT NULL,
    archived_support TEXT NOT NULL,
    semantic_verdict TEXT NOT NULL CHECK (
        semantic_verdict IN (
            'PASS_GRADE_A_PARTIAL_ATTRIBUTION',
            'PASS_GRADE_A_WITH_STATED_ALLOCATION_BOUNDARY',
            'PASS_GRADE_C_DERIVED_ARITHMETIC',
            'PASS_GRADE_C_MECHANISM_ONLY'
        )
    ),
    remaining_boundary TEXT NOT NULL,
    review_provenance TEXT NOT NULL CHECK (
        review_provenance =
            'carried_forward_exactly_from_sealed_v1_nonauthor_review'
    ),
    integration_status TEXT NOT NULL CHECK (
        integration_status = 'accepted_bounded_preservation_rebase'
    )
);

CREATE TABLE private_claim_locator_review_sources (
    claim_locator_review_id TEXT NOT NULL
        REFERENCES private_claim_locator_reviews(claim_locator_review_id),
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    source_order INTEGER NOT NULL CHECK (source_order > 0),
    PRIMARY KEY (claim_locator_review_id, source_id),
    UNIQUE (claim_locator_review_id, source_order)
);

CREATE TABLE private_claim_status_transitions (
    claim_status_rebase_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    prior_review_status TEXT NOT NULL CHECK (
        prior_review_status = 'accepted_bounded_source_archive_hold'
    ),
    proposed_review_status TEXT NOT NULL CHECK (
        proposed_review_status =
            'accepted_bounded_preservation_cleared_other_holds_retained'
    ),
    cleared_hold TEXT NOT NULL CHECK (
        cleared_hold = 'primary_source_snapshot_hash_and_locator_semantics'
    ),
    authorization_boundary TEXT NOT NULL CHECK (
        authorization_boundary =
            'primary_atomic_rebase_accepted_with_retained_holds'
    ),
    notes TEXT NOT NULL
);

CREATE TABLE private_claim_status_retained_holds (
    claim_status_rebase_id TEXT NOT NULL
        REFERENCES private_claim_status_transitions(claim_status_rebase_id),
    hold_order INTEGER NOT NULL CHECK (hold_order > 0),
    retained_hold TEXT NOT NULL,
    PRIMARY KEY (claim_status_rebase_id, hold_order),
    UNIQUE (claim_status_rebase_id, retained_hold)
);

PRAGMA foreign_keys = ON;

CREATE TABLE source_versions (
    source_version_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    provenance_population TEXT NOT NULL,
    original_url TEXT,
    final_resolved_url TEXT,
    retrieval_timestamp_utc TEXT,
    publication_date TEXT,
    preservation_class TEXT NOT NULL CHECK (
        preservation_class IN ('P1','P1_public_primary_document','P3','P4')
    ),
    preservation_status TEXT NOT NULL CHECK (
        preservation_status IN (
            'internal_P4_reconstruction',
            'metadata_only_rights_boundary',
            'preserved_complete',
            'retrieval_failed'
        )
    ),
    mime_type TEXT,
    byte_count INTEGER CHECK (byte_count >= 0),
    content_sha256 TEXT CHECK (
        content_sha256 IS NULL OR length(content_sha256) = 64
    ),
    local_archive_path TEXT,
    preservation_method TEXT,
    response_metadata_state TEXT,
    access_licensing_boundary TEXT,
    artifact_role TEXT CHECK (
        artifact_role IS NULL OR artifact_role IN (
            'internal_derivation_manifest',
            'metadata_only',
            'none',
            'original_governing_bytes'
        )
    ),
    cited_locator_relationship TEXT CHECK (
        cited_locator_relationship IS NULL OR
        cited_locator_relationship IN (
            'carried_forward_from_sealed_nonauthor_claim_review',
            'method_support',
            'pending_semantic_locator_review'
        )
    ),
    exception_reason TEXT,
    origin_candidate_integration_status TEXT NOT NULL,
    integration_status TEXT NOT NULL CHECK (
        integration_status = 'accepted_source_version_with_semantic_holds'
    )
);

CREATE TABLE source_derivatives (
    derivative_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    parent_source_version_id TEXT NOT NULL
        REFERENCES source_versions(source_version_id),
    derivation_method_version TEXT NOT NULL,
    normalization_rule TEXT,
    derivative_sha256 TEXT NOT NULL CHECK (length(derivative_sha256) = 64),
    derivative_byte_count INTEGER NOT NULL CHECK (derivative_byte_count >= 0),
    derivative_path TEXT NOT NULL,
    locator_aid_status TEXT NOT NULL CHECK (
        locator_aid_status = 'locator_aid_only'
    ),
    governing_boundary TEXT NOT NULL CHECK (
        governing_boundary =
            'original_preserved_inline_xbrl_bytes_govern'
    ),
    origin_candidate_integration_status TEXT NOT NULL,
    integration_status TEXT NOT NULL CHECK (
        integration_status = 'accepted_derivative_locator_aid_with_holds'
    )
);

CREATE TABLE source_exceptions (
    exception_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    source_version_id TEXT NOT NULL
        REFERENCES source_versions(source_version_id),
    exception_type TEXT NOT NULL CHECK (
        exception_type IN (
            'metadata_only_rights_boundary',
            'retrieval_failed'
        )
    ),
    exception_reason TEXT NOT NULL,
    content_sha256 TEXT CHECK (
        content_sha256 IS NULL OR length(content_sha256) = 64
    ),
    local_archive_path TEXT,
    resolution_state TEXT NOT NULL CHECK (
        resolution_state = 'open_explicit_hold'
    ),
    publication_effect TEXT NOT NULL CHECK (
        publication_effect =
            'claim_level_support_remains_held_pending_resolution'
    ),
    integration_status TEXT NOT NULL CHECK (
        integration_status = 'accepted_explicit_source_hold'
    )
);

CREATE TABLE source_duplicate_blob_reconciliations (
    source_id TEXT PRIMARY KEY REFERENCES sources(source_id),
    private_source_version_id TEXT NOT NULL
        REFERENCES source_versions(source_version_id),
    chapters_06_09_source_version_id TEXT NOT NULL
        REFERENCES source_versions(source_version_id),
    private_sha256 TEXT NOT NULL CHECK (length(private_sha256) = 64),
    chapters_06_09_sha256 TEXT NOT NULL
        CHECK (length(chapters_06_09_sha256) = 64),
    private_byte_count INTEGER NOT NULL CHECK (private_byte_count >= 0),
    chapters_06_09_byte_count INTEGER NOT NULL
        CHECK (chapters_06_09_byte_count >= 0),
    reconciliation_result TEXT NOT NULL,
    version_policy TEXT NOT NULL
);

CREATE TABLE claim_locator_states (
    claim_locator_state_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    chapter_id TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    locator TEXT NOT NULL,
    support_type TEXT NOT NULL CHECK (
        support_type IN ('corroborating','direct','method')
    ),
    semantic_review_state TEXT NOT NULL CHECK (
        semantic_review_state IN (
            'pending_semantic_locator_review',
            'preserved_and_semantically_reviewed'
        )
    ),
    semantic_verdict TEXT CHECK (
        semantic_verdict IS NULL OR semantic_verdict IN (
            'PASS_GRADE_A_PARTIAL_ATTRIBUTION',
            'PASS_GRADE_A_WITH_STATED_ALLOCATION_BOUNDARY',
            'PASS_GRADE_C_DERIVED_ARITHMETIC',
            'PASS_GRADE_C_MECHANISM_ONLY'
        )
    ),
    remaining_boundary TEXT NOT NULL,
    review_provenance TEXT NOT NULL,
    integration_status TEXT NOT NULL CHECK (
        integration_status = 'accepted_locator_state_with_recorded_boundary'
    )
);

CREATE TABLE chapter_support_state_projection (
    projection_row_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    chapter_id TEXT NOT NULL,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    locator TEXT NOT NULL,
    support_type TEXT NOT NULL CHECK (
        support_type IN (
            'context','contradicting','corroborating','direct','method'
        )
    ),
    support_origin TEXT NOT NULL,
    source_version_id TEXT REFERENCES source_versions(source_version_id),
    source_state_disposition TEXT NOT NULL CHECK (
        source_state_disposition IN (
            'internal_P4_reconstruction',
            'metadata_only_rights_boundary',
            'no_matching_preservation_record',
            'preserved_and_semantically_reviewed',
            'preserved_bytes_available_locator_review_pending',
            'retrieval_failed'
        )
    ),
    semantic_review_authority TEXT NOT NULL CHECK (
        semantic_review_authority IN (
            'accepted_private_v2_nonauthor_review',
            'not_semantically_cleared_by_union_candidate'
        )
    ),
    origin_candidate_integration_status TEXT NOT NULL,
    integration_status TEXT NOT NULL CHECK (
        integration_status = 'accepted_projection_not_publication_clearance'
    )
);

CREATE TABLE source_version_delta_audit (
    delta_audit_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    historical_source_version_id TEXT NOT NULL
        REFERENCES source_versions(source_version_id),
    appended_source_version_id TEXT NOT NULL
        REFERENCES source_versions(source_version_id),
    historical_status TEXT NOT NULL,
    appended_status TEXT NOT NULL,
    alternate_venue_relationship TEXT NOT NULL,
    current_projection_version_id TEXT NOT NULL
        REFERENCES source_versions(source_version_id),
    semantic_locator_state TEXT NOT NULL,
    quarantine_excluded INTEGER NOT NULL CHECK (quarantine_excluded IN (0,1)),
    result TEXT NOT NULL
);

PRAGMA foreign_keys = ON;

CREATE TABLE research_packet_bindings (
    input_domain TEXT PRIMARY KEY,
    author_packet TEXT NOT NULL,
    author_manifest_sha256 TEXT NOT NULL
        CHECK (length(author_manifest_sha256) = 64),
    review_packet TEXT NOT NULL,
    review_manifest_sha256 TEXT NOT NULL
        CHECK (length(review_manifest_sha256) = 64),
    binding_status TEXT NOT NULL CHECK (
        binding_status = 'manifest_bound_ready_input'
    ),
    retained_boundary TEXT NOT NULL,
    authority_class TEXT NOT NULL CHECK (
        authority_class = 'governed_research_archive'
    ),
    economic_promotion_status TEXT NOT NULL CHECK (
        economic_promotion_status = 'not_promoted'
    )
);

CREATE TABLE research_artifacts (
    artifact_id TEXT PRIMARY KEY,
    input_domain TEXT NOT NULL
        REFERENCES research_packet_bindings(input_domain),
    relative_path TEXT NOT NULL,
    byte_count INTEGER NOT NULL CHECK (byte_count >= 0),
    content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
    manifest_rows INTEGER CHECK (manifest_rows >= 0),
    source_load_state TEXT NOT NULL,
    authority_class TEXT NOT NULL CHECK (
        authority_class = 'governed_research_artifact'
    ),
    economic_promotion_status TEXT NOT NULL CHECK (
        economic_promotion_status = 'not_promoted'
    ),
    canonical_relationship_status TEXT NOT NULL CHECK (
        canonical_relationship_status = 'not_asserted'
    ),
    UNIQUE (input_domain, relative_path)
);

CREATE TABLE research_packet_manifest_bindings (
    input_domain TEXT NOT NULL
        REFERENCES research_packet_bindings(input_domain),
    manifest_role TEXT NOT NULL CHECK (
        manifest_role IN ('author','independent_review')
    ),
    relative_path TEXT NOT NULL,
    content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
    PRIMARY KEY (input_domain, manifest_role)
);
