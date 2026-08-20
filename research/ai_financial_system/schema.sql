PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    title TEXT NOT NULL,
    publisher TEXT NOT NULL,
    source_type TEXT NOT NULL,
    url TEXT,
    publication_date TEXT,
    retrieved_at TEXT NOT NULL,
    primary_document INTEGER NOT NULL CHECK (primary_document IN (0,1)),
    archived_path TEXT,
    content_sha256 TEXT,
    evidentiary_scope TEXT NOT NULL,
    limitations TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    jurisdiction TEXT,
    public_private_status TEXT,
    cik TEXT,
    lei TEXT,
    parent_entity_id TEXT REFERENCES entities(entity_id),
    status TEXT NOT NULL DEFAULT 'active',
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    chapter_id TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    confidence TEXT NOT NULL CHECK (
        confidence IN ('verified','high','medium','low','unresolved')
    ),
    claim_type TEXT NOT NULL,
    accounting_status TEXT NOT NULL,
    economic_status TEXT NOT NULL,
    quantification_status TEXT NOT NULL,
    effective_date TEXT,
    supersedes_claim_id TEXT REFERENCES claims(claim_id),
    review_status TEXT NOT NULL DEFAULT 'draft',
    reviewer_notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS evidence_links (
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    locator TEXT NOT NULL,
    support_type TEXT NOT NULL CHECK (
        support_type IN ('direct','corroborating','contradicting','context','method')
    ),
    excerpt_note TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (claim_id, source_id, locator)
);

CREATE TABLE IF NOT EXISTS financial_flows (
    flow_id TEXT PRIMARY KEY,
    from_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    to_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    flow_type TEXT NOT NULL,
    instrument TEXT,
    amount REAL,
    currency TEXT,
    amount_status TEXT NOT NULL,
    announced_date TEXT,
    committed_date TEXT,
    funded_date TEXT,
    recognized_period TEXT,
    recourse TEXT,
    seniority TEXT,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS capital_observations (
    observation_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    reporting_period TEXT NOT NULL,
    measure TEXT NOT NULL,
    amount_low REAL,
    amount_high REAL,
    currency TEXT NOT NULL,
    amount_status TEXT NOT NULL,
    accounting_scope TEXT NOT NULL,
    economic_scope TEXT NOT NULL,
    overlap_group TEXT,
    cash_or_noncash TEXT NOT NULL,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS capital_bridge (
    bridge_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    as_of_period TEXT NOT NULL,
    capital_channel TEXT NOT NULL,
    measure_name TEXT NOT NULL,
    amount_low REAL,
    amount_high REAL,
    currency TEXT NOT NULL,
    temporal_basis TEXT NOT NULL,
    measurement_class TEXT NOT NULL,
    recognition_state TEXT NOT NULL,
    ai_attribution TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS contracts (
    contract_id TEXT PRIMARY KEY,
    party_a_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    party_b_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    contract_type TEXT NOT NULL,
    announcement_date TEXT,
    execution_date TEXT,
    effective_date TEXT,
    expiry_date TEXT,
    commitment_status TEXT NOT NULL,
    stated_value REAL,
    currency TEXT,
    minimum_commitment REAL,
    termination_rights TEXT,
    exclusivity TEXT,
    guarantees TEXT,
    collateral_or_security TEXT,
    disclosed_accounting_treatment TEXT,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS debt_instruments (
    debt_id TEXT PRIMARY KEY,
    issuer_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    lender_or_holder_entity_id TEXT REFERENCES entities(entity_id),
    instrument_type TEXT NOT NULL,
    principal REAL,
    currency TEXT,
    issue_date TEXT,
    maturity_date TEXT,
    coupon_or_rate TEXT,
    seniority TEXT,
    secured_status TEXT,
    guarantor_entity_id TEXT REFERENCES entities(entity_id),
    drawn_amount REAL,
    amount_status TEXT NOT NULL,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    lessee_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    lessor_entity_id TEXT REFERENCES entities(entity_id),
    asset_or_site TEXT NOT NULL,
    lease_type TEXT,
    commencement_date TEXT,
    expiry_date TEXT,
    disclosed_commitment REAL,
    currency TEXT,
    recognized_liability REAL,
    renewal_or_termination_terms TEXT,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS guarantees (
    guarantee_id TEXT PRIMARY KEY,
    guarantor_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    beneficiary_entity_id TEXT REFERENCES entities(entity_id),
    supported_entity_id TEXT REFERENCES entities(entity_id),
    guarantee_type TEXT NOT NULL,
    maximum_exposure REAL,
    currency TEXT,
    effective_date TEXT,
    expiry_date TEXT,
    recognized_liability REAL,
    contingent_status TEXT NOT NULL,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS investments (
    investment_id TEXT PRIMARY KEY,
    investor_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    investee_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    instrument_type TEXT NOT NULL,
    announced_amount REAL,
    funded_amount REAL,
    currency TEXT,
    announcement_date TEXT,
    closing_date TEXT,
    ownership_percentage REAL,
    valuation_basis TEXT,
    governance_rights TEXT,
    strategic_conditions TEXT,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS funding_timeline (
    funding_event_id TEXT PRIMARY KEY,
    recipient_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    investor_entity_id TEXT REFERENCES entities(entity_id),
    event_date TEXT NOT NULL,
    round_or_program TEXT NOT NULL,
    event_type TEXT NOT NULL,
    instrument_type TEXT NOT NULL,
    announced_amount REAL,
    committed_amount REAL,
    funded_amount REAL,
    currency TEXT NOT NULL,
    pre_money_valuation REAL,
    post_money_valuation REAL,
    funding_status TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    supersedes_event_id TEXT REFERENCES funding_timeline(funding_event_id),
    financing_source TEXT,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    project_type TEXT NOT NULL,
    owner_entity_id TEXT REFERENCES entities(entity_id),
    operator_entity_id TEXT REFERENCES entities(entity_id),
    location TEXT,
    announced_capacity TEXT,
    contracted_capacity TEXT,
    operational_capacity TEXT,
    announced_cost REAL,
    funded_cost REAL,
    currency TEXT,
    development_status TEXT NOT NULL,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS project_finance_cases (
    project_finance_case_id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(project_id),
    borrower_entity_id TEXT REFERENCES entities(entity_id),
    sponsor_entity_id TEXT REFERENCES entities(entity_id),
    asset_owner_entity_id TEXT REFERENCES entities(entity_id),
    developer_entity_id TEXT REFERENCES entities(entity_id),
    operator_entity_id TEXT REFERENCES entities(entity_id),
    tenant_entity_id TEXT REFERENCES entities(entity_id),
    end_user_entity_id TEXT REFERENCES entities(entity_id),
    guarantor_entity_id TEXT REFERENCES entities(entity_id),
    project_cost_low REAL,
    project_cost_high REAL,
    debt_committed REAL,
    debt_funded REAL,
    sponsor_equity_disclosed REAL,
    reimbursed_equity REAL,
    currency TEXT,
    loan_to_cost TEXT,
    debt_type TEXT,
    coupon_or_rate TEXT,
    issue_date TEXT,
    maturity_date TEXT,
    amortization_start TEXT,
    construction_status TEXT NOT NULL,
    initial_delivery_date TEXT,
    base_contract_revenue REAL,
    optional_contract_revenue REAL,
    average_annual_noi REAL,
    revenue_model TEXT,
    collateral_or_security TEXT,
    recourse TEXT,
    coverage_or_distribution_test TEXT,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS project_cash_waterfalls (
    waterfall_id TEXT PRIMARY KEY,
    project_finance_case_id TEXT NOT NULL REFERENCES project_finance_cases(project_finance_case_id),
    stage_order INTEGER NOT NULL,
    stage_name TEXT NOT NULL,
    cash_source TEXT NOT NULL,
    payee_or_account TEXT NOT NULL,
    trigger_or_timing TEXT,
    restriction_or_remedy TEXT,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS project_delivery_milestones (
    milestone_id TEXT PRIMARY KEY,
    project_finance_case_id TEXT NOT NULL REFERENCES project_finance_cases(project_finance_case_id),
    milestone_order INTEGER NOT NULL,
    milestone_type TEXT NOT NULL,
    milestone_date TEXT,
    date_status TEXT NOT NULL CHECK (date_status IN ('actual','scheduled','contractual_deadline','derived','undisclosed')),
    completion_status TEXT NOT NULL,
    dependency_or_condition TEXT NOT NULL,
    responsible_or_controlling_party_entity_id TEXT REFERENCES entities(entity_id),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS power_agreements (
    power_id TEXT PRIMARY KEY,
    buyer_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    seller_entity_id TEXT REFERENCES entities(entity_id),
    project_id TEXT REFERENCES projects(project_id),
    agreement_type TEXT NOT NULL,
    capacity_mw REAL,
    energy_mwh REAL,
    term_years REAL,
    execution_date TEXT,
    commencement_date TEXT,
    regulatory_docket TEXT,
    cost_recovery_party TEXT,
    commitment_status TEXT NOT NULL,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS power_system_cases (
    power_system_case_id TEXT PRIMARY KEY,
    case_name TEXT NOT NULL,
    utility_entity_id TEXT REFERENCES entities(entity_id),
    customer_entity_id TEXT REFERENCES entities(entity_id),
    parent_support_entity_id TEXT REFERENCES entities(entity_id),
    project_id TEXT REFERENCES projects(project_id),
    announced_load_mw REAL,
    contracted_load_mw REAL,
    minimum_billing_demand_percent REAL,
    generation_nameplate_mw REAL,
    storage_power_mw REAL,
    storage_energy_mwh REAL,
    transmission_miles REAL,
    base_service_term_years REAL,
    regulatory_docket TEXT,
    service_status TEXT NOT NULL,
    customer_funding_mechanism TEXT,
    utility_funding_mechanism TEXT,
    collateral_or_credit_support TEXT,
    curtailment_or_load_control TEXT,
    residual_cost_holder TEXT,
    accounting_or_ownership TEXT,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS power_infrastructure_assets (
    power_asset_id TEXT PRIMARY KEY,
    power_system_case_id TEXT NOT NULL REFERENCES power_system_cases(power_system_case_id),
    asset_name TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    owner_entity_id TEXT REFERENCES entities(entity_id),
    nameplate_capacity REAL,
    capacity_unit TEXT,
    accredited_capacity REAL,
    storage_duration_hours REAL,
    line_miles_low REAL,
    line_miles_high REAL,
    voltage_kv REAL,
    announced_date TEXT,
    approval_date TEXT,
    construction_start TEXT,
    target_in_service_date TEXT,
    actual_in_service_date TEXT,
    development_status TEXT NOT NULL,
    disclosed_cost REAL,
    currency TEXT,
    funding_mechanism TEXT,
    cost_recovery_mechanism TEXT,
    post_contract_residual TEXT,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS power_cost_waterfalls (
    power_waterfall_id TEXT PRIMARY KEY,
    power_system_case_id TEXT NOT NULL REFERENCES power_system_cases(power_system_case_id),
    stage_order INTEGER NOT NULL,
    stage_name TEXT NOT NULL,
    payer_or_source TEXT NOT NULL,
    payee_or_recovery_account TEXT NOT NULL,
    trigger_or_period TEXT,
    residual_or_remedy TEXT,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS laboratory_control (
    lab_control_id TEXT PRIMARY KEY,
    laboratory_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    controlled_or_related_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    controller_or_counterparty_entity_id TEXT REFERENCES entities(entity_id),
    relationship_type TEXT NOT NULL,
    economic_percentage REAL,
    voting_percentage REAL,
    control_or_governance_rights TEXT NOT NULL,
    effective_date TEXT,
    accounting_perimeter TEXT NOT NULL,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS laboratory_operating_observations (
    lab_observation_id TEXT PRIMARY KEY,
    laboratory_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    observation_date TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    metric_unit TEXT,
    measurement_basis TEXT NOT NULL,
    external_or_related_party TEXT NOT NULL,
    recognized_or_run_rate TEXT NOT NULL,
    reporting_perimeter TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS laboratory_counterparty_flows (
    lab_flow_id TEXT PRIMARY KEY,
    laboratory_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    counterparty_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    flow_direction TEXT NOT NULL,
    flow_type TEXT NOT NULL,
    announced_amount REAL,
    committed_amount REAL,
    funded_amount REAL,
    currency TEXT,
    contract_or_instrument TEXT NOT NULL,
    milestone_or_condition TEXT,
    start_date TEXT,
    end_date TEXT,
    accounting_status TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS hyperscaler_capex_observations (
    hyperscaler_capex_id TEXT PRIMARY KEY,
    company_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    period_start TEXT,
    period_end TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    amount REAL,
    currency TEXT,
    cash_or_noncash TEXT NOT NULL,
    reporting_perimeter TEXT NOT NULL,
    ai_attribution TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS hyperscaler_obligations (
    hyperscaler_obligation_id TEXT PRIMARY KEY,
    company_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    measurement_date TEXT NOT NULL,
    obligation_type TEXT NOT NULL,
    recognized_amount REAL,
    undiscounted_amount REAL,
    commitment_amount REAL,
    currency TEXT,
    commencement_status TEXT NOT NULL,
    accounting_status TEXT NOT NULL,
    maturity_or_term TEXT,
    ai_attribution TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS hyperscaler_revenue_funding (
    hyperscaler_revenue_funding_id TEXT PRIMARY KEY,
    company_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    counterparty_entity_id TEXT REFERENCES entities(entity_id),
    measurement_date TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    amount REAL,
    currency TEXT,
    measurement_basis TEXT NOT NULL,
    cash_status TEXT NOT NULL,
    revenue_recognition_status TEXT NOT NULL,
    reporting_perimeter TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS compute_supply (
    supply_id TEXT PRIMARY KEY,
    supplier_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    customer_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    product_or_service TEXT NOT NULL,
    quantity_text TEXT,
    capacity_measure TEXT,
    delivery_window TEXT,
    contract_status TEXT NOT NULL,
    prepaid_amount REAL,
    minimum_spend REAL,
    currency TEXT,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS compute_commitments (
    compute_commitment_id TEXT PRIMARY KEY,
    supplier_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    customer_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    intermediary_entity_id TEXT REFERENCES entities(entity_id),
    asset_owner_entity_id TEXT REFERENCES entities(entity_id),
    operator_entity_id TEXT REFERENCES entities(entity_id),
    payer_entity_id TEXT REFERENCES entities(entity_id),
    product_or_service TEXT NOT NULL,
    quantity_text TEXT,
    capacity_value REAL,
    capacity_unit TEXT,
    contract_value REAL,
    currency TEXT,
    minimum_spend REAL,
    prepaid_amount REAL,
    announcement_date TEXT,
    execution_date TEXT,
    service_start TEXT,
    expiry_date TEXT,
    status_ladder TEXT NOT NULL,
    binding_scope TEXT NOT NULL,
    termination_rights TEXT,
    renewal_rights TEXT,
    remedies_or_credits TEXT,
    asset_title_status TEXT NOT NULL,
    accounting_treatment TEXT,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS compute_asset_finance (
    asset_finance_id TEXT PRIMARY KEY,
    compute_commitment_id TEXT REFERENCES compute_commitments(compute_commitment_id),
    project_or_asset TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    legal_owner_entity_id TEXT REFERENCES entities(entity_id),
    operator_entity_id TEXT REFERENCES entities(entity_id),
    end_user_entity_id TEXT REFERENCES entities(entity_id),
    financing_provider_entity_id TEXT REFERENCES entities(entity_id),
    instrument_type TEXT NOT NULL,
    headline_amount REAL,
    funded_or_drawn_amount REAL,
    recognized_asset_amount REAL,
    recognized_liability_amount REAL,
    currency TEXT,
    collateral_or_security TEXT,
    recourse TEXT,
    title_or_control_status TEXT NOT NULL,
    recognition_status TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS ownership (
    ownership_id TEXT PRIMARY KEY,
    owner_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    owned_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    instrument_type TEXT NOT NULL,
    economic_percentage REAL,
    voting_percentage REAL,
    as_of_date TEXT,
    control_rights TEXT,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS dependencies (
    dependency_id TEXT PRIMARY KEY,
    dependent_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    provider_entity_id TEXT REFERENCES entities(entity_id),
    dependency_type TEXT NOT NULL,
    input_or_service TEXT NOT NULL,
    concentration_measure TEXT,
    substitution_difficulty TEXT,
    time_to_replace TEXT,
    failure_mode TEXT NOT NULL,
    accounting_visibility TEXT NOT NULL,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS risk_pathways (
    pathway_id TEXT PRIMARY KEY,
    initiating_node TEXT NOT NULL,
    transmission_sequence TEXT NOT NULL,
    affected_entities TEXT NOT NULL,
    balance_sheet_channel TEXT,
    liquidity_channel TEXT,
    operating_channel TEXT,
    control_channel TEXT,
    observable_indicators TEXT NOT NULL,
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    confidence TEXT NOT NULL,
    source_claim_id TEXT REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS semiconductor_physical_dependencies (
    semiconductor_dependency_id TEXT PRIMARY KEY,
    system_controller_entity_id TEXT REFERENCES entities(entity_id),
    dependent_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    provider_entity_id TEXT REFERENCES entities(entity_id),
    provider_name TEXT NOT NULL,
    dependency_layer TEXT NOT NULL,
    input_or_service TEXT NOT NULL,
    documented_role TEXT NOT NULL,
    concentration_measure TEXT,
    qualification_state TEXT NOT NULL,
    volume_state TEXT NOT NULL,
    substitution_state TEXT NOT NULL,
    geographic_locus TEXT,
    failure_mode TEXT NOT NULL,
    accounting_visibility TEXT NOT NULL,
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS semiconductor_capacity_finance (
    capacity_finance_id TEXT PRIMARY KEY,
    recipient_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    provider_entity_id TEXT REFERENCES entities(entity_id),
    provider_name TEXT NOT NULL,
    project_or_capacity TEXT NOT NULL,
    capital_channel TEXT NOT NULL,
    headline_amount REAL,
    funded_or_recognized_amount REAL,
    currency TEXT,
    measurement_or_announcement_date TEXT,
    status_ladder TEXT NOT NULL,
    cash_status TEXT NOT NULL,
    accounting_status TEXT NOT NULL,
    output_milestone TEXT,
    recourse_or_condition TEXT,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS semiconductor_loss_episodes (
    loss_episode_id TEXT PRIMARY KEY,
    episode_status TEXT NOT NULL,
    trigger_date TEXT,
    trigger TEXT NOT NULL,
    affected_product_or_capacity TEXT NOT NULL,
    first_loss_bearing_entity_id TEXT REFERENCES entities(entity_id),
    first_loss_bearing_party TEXT NOT NULL,
    recognized_charge_amount REAL,
    currency TEXT,
    loss_components TEXT NOT NULL,
    transmission_sequence TEXT NOT NULL,
    recovery_or_mitigation TEXT,
    remaining_loss_allocation_unknown TEXT NOT NULL,
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS semiconductor_substitution_constraints (
    substitution_constraint_id TEXT PRIMARY KEY,
    dependent_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    dependency_layer TEXT NOT NULL,
    incumbent_provider_entity_id TEXT REFERENCES entities(entity_id),
    incumbent_provider_name TEXT NOT NULL,
    candidate_alternative TEXT NOT NULL,
    design_portability_evidence TEXT NOT NULL,
    qualification_evidence TEXT NOT NULL,
    commercial_volume_evidence TEXT NOT NULL,
    estimated_time_to_replace TEXT,
    performance_or_yield_penalty TEXT,
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    linked_unknown_id TEXT REFERENCES unknowns(unknown_id),
    current_conclusion TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS semiconductor_regulatory_controls (
    regulatory_control_id TEXT PRIMARY KEY,
    authority TEXT NOT NULL,
    jurisdiction TEXT NOT NULL,
    announcement_or_effective_date TEXT,
    covered_product_or_activity TEXT NOT NULL,
    customer_or_geographic_test TEXT NOT NULL,
    license_posture TEXT NOT NULL,
    physical_control_effect TEXT NOT NULL,
    documented_financial_effect TEXT,
    policy_state TEXT NOT NULL,
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS semiconductor_measurement_reconciliations (
    reconciliation_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    measure_or_statement_a TEXT NOT NULL,
    measure_or_statement_b TEXT NOT NULL,
    apparent_conflict TEXT NOT NULL,
    reconciliation TEXT NOT NULL,
    current_status TEXT NOT NULL,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    primary_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    corroborating_claim_ids TEXT,
    remaining_unknown TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS private_credit_exposures (
    exposure_id TEXT PRIMARY KEY,
    holder_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    borrower_entity_id TEXT REFERENCES entities(entity_id),
    borrower_name TEXT NOT NULL,
    project_or_collateral TEXT NOT NULL,
    instrument_type TEXT NOT NULL,
    lien_or_seniority TEXT,
    rate_or_coupon TEXT,
    maturity_date TEXT,
    principal_or_face REAL,
    cost_basis REAL,
    fair_value REAL,
    unfunded_commitment REAL,
    currency TEXT NOT NULL,
    measurement_date TEXT NOT NULL,
    valuation_level TEXT,
    accrual_status TEXT NOT NULL,
    ai_linkage_basis TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS private_credit_fund_liquidity (
    fund_liquidity_id TEXT PRIMARY KEY,
    fund_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    measurement_date TEXT NOT NULL,
    vehicle_type TEXT NOT NULL,
    public_market_status TEXT NOT NULL,
    net_assets_or_nav REAL,
    gross_assets REAL,
    indebtedness REAL,
    currency TEXT NOT NULL,
    repurchase_limit TEXT NOT NULL,
    repurchase_discretion TEXT NOT NULL,
    period_repurchase_amount REAL,
    distribution_per_share REAL,
    return_of_capital_per_share REAL,
    valuation_basis TEXT NOT NULL,
    liquidity_mismatch_summary TEXT NOT NULL,
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS private_credit_warehouse_controls (
    warehouse_control_id TEXT PRIMARY KEY,
    borrower_spv_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    lender_entity_id TEXT REFERENCES entities(entity_id),
    lender_name TEXT NOT NULL,
    facility_or_tranche TEXT NOT NULL,
    commitment_or_capacity REAL,
    drawn_amount REAL,
    currency TEXT NOT NULL,
    rate_or_spread TEXT,
    maturity_date TEXT,
    collateral_scope TEXT NOT NULL,
    advance_rate TEXT NOT NULL,
    borrowing_base_or_ratio_trigger TEXT,
    cure_period TEXT,
    cash_control TEXT NOT NULL,
    servicer_remedy TEXT NOT NULL,
    disposition_or_step_in_rights TEXT NOT NULL,
    ai_collateral_mapping_status TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS private_credit_valuation_clocks (
    valuation_clock_id TEXT PRIMARY KEY,
    sequence_order INTEGER NOT NULL,
    trigger_or_state TEXT NOT NULL,
    primary_measurement TEXT NOT NULL,
    first_observer_or_controller TEXT NOT NULL,
    potential_action TEXT NOT NULL,
    public_visibility TEXT NOT NULL,
    default_required INTEGER NOT NULL CHECK (default_required IN (0, 1)),
    evidence_basis TEXT NOT NULL,
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS private_credit_measurement_reconciliations (
    reconciliation_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    measure_or_statement_a TEXT NOT NULL,
    measure_or_statement_b TEXT NOT NULL,
    apparent_conflict TEXT NOT NULL,
    reconciliation TEXT NOT NULL,
    current_status TEXT NOT NULL,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    primary_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    corroborating_claim_ids TEXT,
    remaining_unknown TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS public_capital_awards (
    award_id TEXT PRIMARY KEY,
    recipient_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    provider_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    program_name TEXT NOT NULL,
    project_or_capacity TEXT NOT NULL,
    instrument_type TEXT NOT NULL,
    preliminary_amount REAL,
    final_award_ceiling REAL,
    committed_amount REAL,
    disbursed_amount REAL,
    recognized_amount REAL,
    currency TEXT NOT NULL,
    preliminary_date TEXT,
    final_award_date TEXT,
    measurement_date TEXT NOT NULL,
    status_ladder TEXT NOT NULL,
    milestone_conditions TEXT NOT NULL,
    clawback_or_repayment TEXT NOT NULL,
    output_milestone TEXT NOT NULL,
    ai_attribution_basis TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS public_tax_incentives (
    tax_incentive_id TEXT PRIMARY KEY,
    taxpayer_entity_id TEXT REFERENCES entities(entity_id),
    taxpayer_name TEXT NOT NULL,
    authority_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    statute_or_program TEXT NOT NULL,
    project_or_property TEXT NOT NULL,
    statutory_rate TEXT NOT NULL,
    qualified_basis REAL,
    recognized_credit REAL,
    cash_collected_or_offset REAL,
    currency TEXT NOT NULL,
    placed_in_service_rule TEXT NOT NULL,
    measurement_date TEXT NOT NULL,
    accounting_status TEXT NOT NULL,
    election_or_payment_status TEXT NOT NULL,
    recapture_or_disqualification TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS public_equity_positions (
    equity_position_id TEXT PRIMARY KEY,
    holder_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    issuer_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    source_program TEXT NOT NULL,
    cash_consideration REAL,
    shares_issued_or_released REAL,
    shares_in_escrow REAL,
    warrant_shares REAL,
    stated_price_per_share REAL,
    beneficial_ownership_percentage REAL,
    currency TEXT NOT NULL,
    measurement_date TEXT NOT NULL,
    transaction_status TEXT NOT NULL,
    voting_terms TEXT NOT NULL,
    transfer_or_sale_terms TEXT NOT NULL,
    contingent_trigger TEXT NOT NULL,
    issuer_accounting_treatment TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS public_infrastructure_cost_allocation (
    allocation_id TEXT PRIMARY KEY,
    public_or_regulated_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    customer_entity_id TEXT REFERENCES entities(entity_id),
    customer_name TEXT NOT NULL,
    project_or_asset TEXT NOT NULL,
    ownership_and_financing TEXT NOT NULL,
    customer_contribution TEXT NOT NULL,
    contract_recovery TEXT NOT NULL,
    public_or_ratepayer_residual TEXT NOT NULL,
    shared_system_benefit TEXT NOT NULL,
    measurement_date TEXT NOT NULL,
    disclosed_cost REAL,
    currency TEXT,
    cost_status TEXT NOT NULL,
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sovereign_capital_cases (
    sovereign_case_id TEXT PRIMARY KEY,
    sovereign_or_public_controller_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    vehicle_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    investee_or_project_entity_id TEXT REFERENCES entities(entity_id),
    investee_or_project_name TEXT NOT NULL,
    ownership_or_control TEXT NOT NULL,
    headline_or_target_amount REAL,
    committed_amount REAL,
    funded_amount REAL,
    currency TEXT,
    instrument_or_vehicle TEXT NOT NULL,
    measurement_date TEXT NOT NULL,
    status_ladder TEXT NOT NULL,
    strategic_or_location_conditions TEXT NOT NULL,
    disclosure_quality TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS public_capital_measurement_reconciliations (
    reconciliation_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    measure_or_statement_a TEXT NOT NULL,
    measure_or_statement_b TEXT NOT NULL,
    apparent_conflict TEXT NOT NULL,
    reconciliation TEXT NOT NULL,
    current_status TEXT NOT NULL,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    primary_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    corroborating_claim_ids TEXT,
    remaining_unknown TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claim_confidence_history (
    history_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    observed_at TEXT NOT NULL,
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    confidence TEXT NOT NULL,
    prior_evidence_grade TEXT,
    prior_confidence TEXT,
    evidence_change TEXT NOT NULL,
    reason_for_change TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS claim_economic_importance (
    importance_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    magnitude TEXT NOT NULL CHECK (
        magnitude IN ('systemically_consequential','high','medium','low')
    ),
    horizon TEXT NOT NULL CHECK (horizon IN ('immediate','structural','optional')),
    valuation_or_risk_channel TEXT NOT NULL,
    rationale TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS economic_exposures (
    exposure_id TEXT PRIMARY KEY,
    exposure_name TEXT NOT NULL,
    exposure_type TEXT NOT NULL,
    legal_obligor_entity_id TEXT REFERENCES entities(entity_id),
    economic_beneficiary_entity_id TEXT REFERENCES entities(entity_id),
    project_id TEXT REFERENCES projects(project_id),
    instrument TEXT NOT NULL,
    amount_low REAL,
    amount_high REAL,
    currency TEXT,
    amount_type TEXT NOT NULL CHECK (
        amount_type IN ('stock','flow','ceiling','floor','contingent','physical_quantity','unquantified')
    ),
    funding_status TEXT NOT NULL,
    recognition_status TEXT NOT NULL,
    recourse TEXT,
    seniority TEXT,
    maturity_date TEXT,
    cancellation_rights TEXT,
    guarantee_id TEXT REFERENCES guarantees(guarantee_id),
    first_loss_bearer_entity_id TEXT REFERENCES entities(entity_id),
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    economic_importance TEXT NOT NULL CHECK (
        economic_importance IN ('systemically_consequential','high','medium','low')
    ),
    confidence TEXT NOT NULL CHECK (
        confidence IN ('verified','high','medium','low','unresolved')
    ),
    falsification_condition TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS exposure_components (
    exposure_id TEXT NOT NULL REFERENCES economic_exposures(exposure_id),
    component_type TEXT NOT NULL,
    component_id TEXT NOT NULL,
    observation_role TEXT NOT NULL,
    overlap_group TEXT,
    additive_eligible INTEGER NOT NULL CHECK (additive_eligible IN (0, 1)),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (exposure_id, component_type, component_id)
);

CREATE TABLE IF NOT EXISTS utilization_observations (
    utilization_observation_id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(project_id),
    supplier_entity_id TEXT REFERENCES entities(entity_id),
    customer_entity_id TEXT REFERENCES entities(entity_id),
    asset_or_service TEXT NOT NULL,
    period_start TEXT,
    period_end TEXT,
    quantity_unit TEXT NOT NULL,
    installed_quantity REAL,
    available_quantity REAL,
    announced_quantity REAL,
    binding_base_quantity REAL,
    optional_expansion_quantity REAL,
    delivered_quantity REAL,
    energized_quantity REAL,
    accepted_quantity REAL,
    reserved_quantity REAL,
    consumed_quantity REAL,
    billable_quantity REAL,
    productive_quantity REAL,
    realized_revenue REAL,
    revenue_currency TEXT,
    realized_unit_price REAL,
    financial_demand_status TEXT NOT NULL,
    physical_demand_status TEXT NOT NULL,
    economic_demand_status TEXT NOT NULL,
    evidence_grade TEXT NOT NULL CHECK (evidence_grade IN ('A','B','C','D')),
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS chapter_handoffs (
    handoff_id TEXT PRIMARY KEY,
    upstream_chapter_id TEXT NOT NULL,
    downstream_chapter_id TEXT NOT NULL,
    dataset_or_ledger TEXT NOT NULL,
    analytical_input TEXT NOT NULL,
    required_fields TEXT NOT NULL,
    readiness_status TEXT NOT NULL CHECK (
        readiness_status IN ('ready','bounded','blocked','not_started')
    ),
    missing_evidence TEXT NOT NULL,
    source_claim_id TEXT REFERENCES claims(claim_id),
    as_of_date TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS unknowns (
    unknown_id TEXT PRIMARY KEY,
    chapter_id TEXT NOT NULL,
    question TEXT NOT NULL,
    materiality TEXT NOT NULL,
    unknown_class TEXT NOT NULL DEFAULT 'disclosure_gap',
    downstream_blocking_chapters TEXT NOT NULL DEFAULT '',
    resolution_trigger TEXT NOT NULL DEFAULT '',
    suspected_sources TEXT NOT NULL,
    resolution_plan TEXT NOT NULL,
    status TEXT NOT NULL,
    last_reviewed TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_chapter ON claims(chapter_id);
CREATE INDEX IF NOT EXISTS idx_claims_grade ON claims(evidence_grade);
CREATE INDEX IF NOT EXISTS idx_capital_observations_entity
    ON capital_observations(entity_id, reporting_period);
CREATE INDEX IF NOT EXISTS idx_flows_entities ON financial_flows(from_entity_id, to_entity_id);
CREATE INDEX IF NOT EXISTS idx_compute_commitment_parties
ON compute_commitments(supplier_entity_id, customer_entity_id);
CREATE INDEX IF NOT EXISTS idx_laboratory_control_entities
ON laboratory_control(laboratory_entity_id, controller_or_counterparty_entity_id);
CREATE INDEX IF NOT EXISTS idx_laboratory_flows
ON laboratory_counterparty_flows(laboratory_entity_id, counterparty_entity_id);
CREATE INDEX IF NOT EXISTS idx_hyperscaler_capex
ON hyperscaler_capex_observations(company_entity_id, period_end);
CREATE INDEX IF NOT EXISTS idx_hyperscaler_obligations
ON hyperscaler_obligations(company_entity_id, measurement_date);
CREATE INDEX IF NOT EXISTS idx_hyperscaler_revenue_funding
ON hyperscaler_revenue_funding(company_entity_id, counterparty_entity_id);
CREATE INDEX IF NOT EXISTS idx_compute_asset_finance_parties
ON compute_asset_finance(legal_owner_entity_id, operator_entity_id, end_user_entity_id);
CREATE INDEX IF NOT EXISTS idx_contracts_parties ON contracts(party_a_entity_id, party_b_entity_id);
CREATE INDEX IF NOT EXISTS idx_project_finance_parties
ON project_finance_cases(borrower_entity_id, sponsor_entity_id, tenant_entity_id);
CREATE INDEX IF NOT EXISTS idx_dependencies_entities
    ON dependencies(dependent_entity_id, provider_entity_id);
CREATE INDEX IF NOT EXISTS idx_semiconductor_dependencies
    ON semiconductor_physical_dependencies(dependent_entity_id, provider_entity_id);
CREATE INDEX IF NOT EXISTS idx_semiconductor_capacity_finance
    ON semiconductor_capacity_finance(recipient_entity_id, provider_entity_id);
CREATE INDEX IF NOT EXISTS idx_semiconductor_loss_entity
    ON semiconductor_loss_episodes(first_loss_bearing_entity_id);
CREATE INDEX IF NOT EXISTS idx_semiconductor_substitution
    ON semiconductor_substitution_constraints(dependent_entity_id, incumbent_provider_entity_id);
CREATE INDEX IF NOT EXISTS idx_semiconductor_regulatory_controls
    ON semiconductor_regulatory_controls(announcement_or_effective_date, policy_state);
CREATE INDEX IF NOT EXISTS idx_semiconductor_reconciliations
    ON semiconductor_measurement_reconciliations(topic, current_status);
CREATE INDEX IF NOT EXISTS idx_private_credit_exposure_parties
    ON private_credit_exposures(holder_entity_id, borrower_entity_id);
CREATE INDEX IF NOT EXISTS idx_private_credit_fund_liquidity
    ON private_credit_fund_liquidity(fund_entity_id, measurement_date);
CREATE INDEX IF NOT EXISTS idx_private_credit_warehouse
    ON private_credit_warehouse_controls(borrower_spv_entity_id, lender_entity_id);
CREATE INDEX IF NOT EXISTS idx_private_credit_valuation_clocks
    ON private_credit_valuation_clocks(sequence_order);
CREATE INDEX IF NOT EXISTS idx_private_credit_reconciliations
    ON private_credit_measurement_reconciliations(topic, current_status);
CREATE INDEX IF NOT EXISTS idx_public_awards_parties
    ON public_capital_awards(recipient_entity_id, provider_entity_id);
CREATE INDEX IF NOT EXISTS idx_public_tax_taxpayer
    ON public_tax_incentives(taxpayer_entity_id, measurement_date);
CREATE INDEX IF NOT EXISTS idx_public_equity_parties
    ON public_equity_positions(holder_entity_id, issuer_entity_id);
CREATE INDEX IF NOT EXISTS idx_public_infrastructure_parties
    ON public_infrastructure_cost_allocation(public_or_regulated_entity_id, customer_entity_id);
CREATE INDEX IF NOT EXISTS idx_sovereign_capital_parties
    ON sovereign_capital_cases(sovereign_or_public_controller_entity_id, vehicle_entity_id);
CREATE INDEX IF NOT EXISTS idx_public_capital_reconciliations
    ON public_capital_measurement_reconciliations(topic, current_status);
CREATE INDEX IF NOT EXISTS idx_claim_confidence_history
    ON claim_confidence_history(claim_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_claim_economic_importance
    ON claim_economic_importance(claim_id, magnitude, horizon);
CREATE INDEX IF NOT EXISTS idx_economic_exposure_parties
    ON economic_exposures(legal_obligor_entity_id, economic_beneficiary_entity_id);
CREATE INDEX IF NOT EXISTS idx_economic_exposure_project
    ON economic_exposures(project_id, exposure_type);
CREATE INDEX IF NOT EXISTS idx_exposure_components_identity
    ON exposure_components(component_type, component_id);
CREATE INDEX IF NOT EXISTS idx_utilization_project_period
    ON utilization_observations(project_id, period_end);
CREATE INDEX IF NOT EXISTS idx_utilization_parties
    ON utilization_observations(supplier_entity_id, customer_entity_id);
CREATE INDEX IF NOT EXISTS idx_chapter_handoffs
    ON chapter_handoffs(upstream_chapter_id, downstream_chapter_id, readiness_status);
