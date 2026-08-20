-- Chapter 16 rebuilt-v2 lossless staging schema. DESIGN ONLY.
-- No canonical database write is authorized by this packet.
CREATE TABLE chapter16_relationship_candidate_universe (
  candidate_id TEXT PRIMARY KEY,
  candidate_source_table TEXT,
  candidate_source_record_id TEXT,
  candidate_origin TEXT,
  from_party_id TEXT,
  to_party_id TEXT,
  project_id TEXT,
  from_role TEXT,
  to_role TEXT,
  relationship_type TEXT,
  amount_state TEXT,
  nonadditivity_rule TEXT,
  evidence_state TEXT,
  evidence_grade TEXT,
  source_claim_id TEXT,
  as_of_date TEXT,
  eligibility_status TEXT,
  selection_decision TEXT,
  selection_reason TEXT,
  edge_id_if_included TEXT,
  universe_completeness_status TEXT,
  adjudication_state TEXT,
  graph_selection_state TEXT,
  relationship_equivalence_status TEXT
);

CREATE TABLE chapter16_counterparty_graph_selection_policy (
  policy_id TEXT PRIMARY KEY,
  policy_domain TEXT,
  selection_action TEXT,
  rule TEXT,
  deterministic_test TEXT,
  publication_effect TEXT,
  source_claim_id TEXT
);

CREATE TABLE chapter16_graph_node_registry (
  node_id TEXT PRIMARY KEY,
  node_label TEXT,
  node_class TEXT,
  canonical_status TEXT,
  canonical_party_node_included TEXT,
  one_to_one_legal_entity TEXT,
  party_resolution_status TEXT,
  economic_group_id TEXT,
  economic_group_label TEXT,
  group_mapping_basis TEXT,
  mapping_evidence_state TEXT,
  mapping_evidence_grade TEXT,
  source_claim_id TEXT,
  cohort_multiplicity TEXT,
  wrapper_status TEXT,
  project_status TEXT,
  notes TEXT
);

CREATE TABLE chapter16_transaction_network_edges (
  edge_id TEXT PRIMARY KEY,
  candidate_id TEXT,
  from_entity_id TEXT,
  to_entity_id TEXT,
  project_id TEXT,
  from_role TEXT,
  to_role TEXT,
  edge_type TEXT,
  amount_state TEXT,
  edge_weight TEXT,
  nonadditivity_rule TEXT,
  evidence_state TEXT,
  evidence_grade TEXT,
  source_claim_id TEXT,
  as_of_date TEXT,
  notes TEXT
);

CREATE TABLE chapter16_graph_edge_selection_audit (
  candidate_id TEXT PRIMARY KEY,
  edge_id TEXT,
  selection_action TEXT,
  canonical_party_pair_key TEXT,
  economic_group_pair_key TEXT,
  canonical_party_incidence_use TEXT,
  canonical_party_degree_use TEXT,
  economic_group_incidence_use TEXT,
  economic_group_degree_use TEXT,
  weight_rule TEXT,
  amount_rule TEXT,
  source_claim_id TEXT,
  selection_reason TEXT
);

CREATE TABLE chapter16_entity_centrality_snapshot (
  snapshot_id TEXT PRIMARY KEY,
  view TEXT,
  node_id TEXT,
  node_label TEXT,
  unique_counterparties TEXT,
  selected_population TEXT,
  selected_evidence_connectivity TEXT,
  incident_role_edge_count TEXT,
  formula TEXT,
  role_edge_population TEXT,
  unique_pair_population TEXT,
  selection_boundary TEXT,
  weight_rule TEXT,
  evidence_state TEXT,
  evidence_grade TEXT,
  source_claim_id TEXT,
  notes TEXT
);

CREATE TABLE chapter16_project_party_registry (
  project_party_registry_id TEXT PRIMARY KEY,
  party_id TEXT,
  party_label TEXT,
  party_type TEXT,
  canonical_entity_id TEXT,
  multiplicity TEXT,
  source_claim_id TEXT,
  identity_status TEXT,
  notes TEXT
);

CREATE TABLE chapter16_project_role_edges (
  project_edge_id TEXT PRIMARY KEY,
  project_id TEXT,
  party_id TEXT,
  party_registry_id TEXT,
  role TEXT,
  source_claim_id TEXT,
  as_of_date TEXT,
  nonadditivity_note TEXT
);

CREATE TABLE chapter16_project_centrality_snapshot (
  snapshot_id TEXT PRIMARY KEY,
  project_id TEXT,
  view TEXT,
  unique_documented_counterparties TEXT,
  selected_party_or_cohort_population TEXT,
  selected_evidence_connectivity TEXT,
  formula TEXT,
  case_population TEXT,
  role_types TEXT,
  date_scope TEXT,
  identity_state TEXT,
  nonadditivity_control TEXT,
  evidence_state TEXT,
  evidence_grade TEXT,
  source_claim_id TEXT,
  notes TEXT
);

CREATE TABLE chapter16_single_point_failure_register (
  failure_id TEXT PRIMARY KEY,
  dependent_node_or_project TEXT,
  mechanism TEXT,
  documented_lag_or_trigger TEXT,
  documented_mitigant TEXT,
  first_loss_bearer TEXT,
  evidence_state TEXT,
  evidence_grade TEXT,
  source_claim_id TEXT,
  as_of_date TEXT,
  interpretation_limit TEXT
);

CREATE TABLE chapter16_single_point_failure_parties (
  failure_party_id TEXT PRIMARY KEY,
  failure_id TEXT,
  party_id TEXT,
  party_role TEXT,
  source_claim_id TEXT
);

CREATE TABLE chapter16_commensurate_concentration_metrics (
  metric_id TEXT PRIMARY KEY,
  subject_id TEXT,
  population_scope TEXT,
  common_measure_basis TEXT,
  total_or_known_share TEXT,
  metric TEXT,
  metric_value TEXT,
  metric_status TEXT,
  source_claim_id TEXT,
  evidence_state TEXT,
  evidence_grade TEXT,
  nonadditivity_control TEXT,
  interpretation_limit TEXT
);

CREATE TABLE chapter16_customer_revenue_concentration (
  concentration_id TEXT PRIMARY KEY,
  provider_or_project TEXT,
  customer_identity TEXT,
  observed_measure TEXT,
  known_share_or_value TEXT,
  concentration_metric TEXT,
  metric_status TEXT,
  source_claim_id TEXT,
  as_of_date TEXT,
  nonadditivity_rule TEXT,
  evidence_state TEXT,
  evidence_grade TEXT,
  interpretation_limit TEXT
);

CREATE TABLE chapter16_supplier_platform_concentration (
  supplier_concentration_id TEXT PRIMARY KEY,
  dependent_entity_id TEXT,
  supplier_or_platform_entity_id TEXT,
  observed_concentration_measure TEXT,
  measure_value TEXT,
  substitution_state TEXT,
  source_claim_id TEXT,
  as_of_date TEXT,
  evidence_state TEXT,
  evidence_grade TEXT,
  nonadditivity_rule TEXT,
  interpretation_limit TEXT
);

CREATE TABLE chapter16_guarantee_concentration (
  guarantee_concentration_id TEXT PRIMARY KEY,
  guarantor_entity_id TEXT,
  named_or_bounded_relationship_count TEXT,
  maximum_or_threshold_vector TEXT,
  currency TEXT,
  aggregation_status TEXT,
  concentration_interpretation TEXT,
  source_claim_id TEXT,
  as_of_date TEXT,
  nonadditivity_rule TEXT,
  evidence_state TEXT,
  evidence_grade TEXT,
  notes TEXT
);

CREATE TABLE chapter16_identity_sensitivity (
  sensitivity_id TEXT PRIMARY KEY,
  affected_output TEXT,
  unresolved_identity_or_amount TEXT,
  sensitivity_direction TEXT,
  quantitative_effect TEXT,
  current_treatment TEXT,
  source_claim_id TEXT,
  required_resolution TEXT
);

CREATE TABLE chapter16_writing_gate_audit (
  gate_id TEXT PRIMARY KEY,
  publication_statement TEXT,
  required_evidence TEXT,
  packet_status TEXT,
  evidence_grade TEXT,
  source_claim_id TEXT,
  decision TEXT
);

CREATE TABLE chapter16_proposed_projects (
  provisional_project_id TEXT PRIMARY KEY,
  proposed_project_name TEXT,
  project_type TEXT,
  borrower_entity_id TEXT,
  owner_entity_id TEXT,
  operator_entity_id TEXT,
  tenant_entity_id TEXT,
  sponsor_entity_id TEXT,
  canonical_project_finance_case_id TEXT,
  canonical_borrower_entity_join_id TEXT,
  private_credit_exposure_join_id TEXT,
  principal_source_claim_id TEXT,
  supporting_source_claim_ids TEXT,
  identity_status TEXT,
  required_integration_action TEXT,
  notes TEXT
);

CREATE TABLE chapter16_record_claim_links (
  record_claim_link_id TEXT PRIMARY KEY,
  source_csv TEXT,
  record_type TEXT,
  record_id TEXT,
  source_field TEXT,
  claim_id TEXT,
  lineage_role TEXT,
  ordinal TEXT
);
