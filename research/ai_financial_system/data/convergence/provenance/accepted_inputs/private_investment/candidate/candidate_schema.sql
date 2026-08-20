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
    integration_status TEXT NOT NULL,
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
    integration_status TEXT NOT NULL,
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
    integration_status TEXT NOT NULL,
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
    integration_status TEXT NOT NULL,
    notes TEXT NOT NULL
);
