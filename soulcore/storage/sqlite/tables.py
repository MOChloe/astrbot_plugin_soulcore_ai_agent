AI_TASK_INSTANCE_TABLES = ("ai_task_audit", "ai_tasks")
CONTEXT_INSTANCE_TABLES = (
    "inbound_voice_admissions",
    "inbound_recall_receipts",
    "inbound_message_recall_states",
    "message_retraction_actions",
    "instance_message_fragments",
    "group_flow_window_members",
    "group_flow_windows",
    "group_reply_relocation_states",
    "group_flow_instance_state",
    "conversation_turn_buffer_members",
    "conversation_turn_buffer_batches",
    "context_build_reports",
    "dialogue_summaries",
    "instance_messages",
)
FILE_INSTANCE_TABLES = (
    "important_todos",
    "file_assets",
    "file_generation_jobs",
)
KNOWLEDGE_INSTANCE_CLEAR_TABLES = (
    "recall_graph_edges",
    "recall_graph_nodes",
    "recall_scene_members",
    "recall_edges",
    "recall_embeddings",
    "recall_documents_fts",
    "recall_scenes",
    "recall_documents",
    "recall_index_generations",
    "recall_index_outbox",
    "knowledge_audit",
    "recall_probe_reports",
    "memories",
    "knowledge_fact_entries",
    "knowledge_message_marks",
    "knowledge_batches",
    "knowledge_processing_state",
)
BACKGROUND_INSTANCE_TABLES = (
    "background_role_timeline_events",
    "background_story_sources",
    "background_role_current_views",
    "background_author_publications",
    "background_author_states",
    "background_initialization_openings",
    "background_instances",
)
MEDIA_INSTANCE_TABLES = (
    "media_cleanup_events",
    "platform_message_media_refs",
    "media_projections",
    "media_assets",
)
PHASE2_INSTANCE_TABLES = (
    "contact_evidence_reservations",
    "character_intent_events",
    "character_intent_evidence",
    "character_intent_revisions",
    "character_intents",
    "deferred_message_items",
    "deferred_message_batches",
    "instance_state_gate_snapshots",
    "instance_state_gate_overrides",
)
PHASE2_RUNTIME_INSTANCE_TABLES = tuple(
    table
    for table in PHASE2_INSTANCE_TABLES
    if table not in {"instance_state_gate_overrides", "character_intent_revisions"}
)
STICKER_INSTANCE_TABLES = (
    "sticker_instance_item_states",
    "media_retention_holds",
    "sticker_reinforcements",
    "sticker_usages",
    "sticker_run_candidates",
    "sticker_import_events",
    "sticker_fingerprints",
    "sticker_items",
    "sticker_clusters",
    "sticker_candidates",
    "sticker_trigger_states",
)
WEB_RUNTIME_INSTANCE_TABLES = (
    "web_page_snapshots",
    "web_image_search_results",
    "web_search_results",
    "web_search_sessions",
)

# Every table that owns state for one conversation instance, except the stable
# route binding and its reset-in-place Core clock.  The order is child-first so
# destructive reset remains valid under SQLite foreign-key enforcement.
#
# Keep this inventory exhaustive: a schema-contract test fails when a new table
# gains both ``profile_id`` and ``instance_id`` without being classified here.
INSTANCE_RESET_DELETE_TABLES = (
    "recall_graph_edges",
    "recall_graph_nodes",
    "recall_scene_members",
    "recall_edges",
    "recall_embeddings",
    "recall_documents_fts",
    "recall_scenes",
    "recall_documents",
    "recall_index_generations",
    "recall_index_outbox",
    "main_core_work_file_bindings",
    "sticker_intake_entries",
    "sticker_check_revisions",
    "ai_task_attempts",
    "ai_work_events",
    "background_role_timeline_events",
    "background_story_sources",
    "background_role_current_views",
    "background_author_publications",
    "background_initialization_openings",
    "sticker_intake_sessions",
    "sticker_import_events",
    "important_todos",
    "character_intent_evidence",
    "character_intent_events",
    "character_intent_revisions",
    "sticker_instance_item_states",
    "sticker_usages",
    "sticker_run_candidates",
    "sticker_reinforcements",
    "sticker_fingerprints",
    "sticker_candidates",
    "platform_message_media_refs",
    "inbound_voice_admissions",
    "inbound_recall_receipts",
    "message_retraction_actions",
    "memory_revision_sources",
    "memory_terms",
    "memory_revisions",
    "media_retention_holds",
    "media_cleanup_events",
    "media_asset_message_links",
    "media_projections",
    "knowledge_message_marks",
    "knowledge_batch_messages",
    "instance_outbox",
    "expression_interruption_events",
    "deferred_message_items",
    "character_intents",
    "web_page_snapshots",
    "timer_run_refs",
    "timer_lifecycle_reviews",
    "sticker_trigger_states",
    "sticker_items",
    "player_profile_entries",
    "media_assets",
    "main_core_work_recovery_wakes",
    "knowledge_processing_state",
    "knowledge_batches",
    "instance_state_gate_snapshots",
    "instance_expression_batches",
    "group_flow_window_members",
    "file_generation_jobs",
    "deferred_message_batches",
    "ai_task_audit",
    "knowledge_fact_revision_sources",
    "knowledge_fact_terms",
    "knowledge_fact_revisions",
    "web_search_results",
    "web_image_search_results",
    "timer_occurrences",
    "timer_occurrence_rolls",
    "sticker_clusters",
    "player_profile_entry_revisions",
    "player_profile_command_receipts",
    "main_core_work_checkpoint_receipts",
    "main_core_work_checkpoint_events",
    "instance_message_fragments",
    "inbound_message_recall_states",
    "instance_core_runs",
    "group_reply_relocation_states",
    "group_flow_windows",
    "group_flow_instance_state",
    "dialogue_summaries",
    "conversation_turn_buffer_members",
    "ai_tasks",
    "knowledge_fact_entries",
    "web_search_sessions",
    "visual_observation_cache",
    "timer_rules",
    "timer_operation_receipts",
    "sticker_libraries",
    "soulcore_logs",
    "player_profiles",
    "platform_send_permits",
    "memories",
    "main_core_work_checkpoints",
    "recall_probe_reports",
    "knowledge_audit",
    "instance_wakeups",
    "instance_participant_identities",
    "instance_messages",
    "instance_main_core_occupancies",
    "background_author_states",
    "background_instances",
    "instance_delivery_state",
    "instance_contact_state",
    "file_assets",
    "conversation_turn_buffer_batches",
    "context_build_reports",
    "contact_expedite_events",
    "contact_evidence_reservations",
    "contact_attempts",
    "ai_workflows",
)

# These children cannot be filtered by profile/instance directly. Their owning
# parent is deleted by the same explicit reset transaction and enforces cascade.
INSTANCE_RESET_INDIRECT_CASCADE_TABLES = (
    "recall_scene_members",  # recall_scenes / recall_documents
    "recall_embeddings",  # recall_documents / recall_index_generations
    "sticker_intake_entries",  # sticker_intake_sessions
    "sticker_check_revisions",  # sticker_candidates
    "ai_task_attempts",  # ai_tasks
    "ai_work_events",  # ai_workflows
    "character_intent_revisions",  # character_intents
    "memory_terms",  # memories / memory_revisions
    "memory_revisions",  # memories
    "media_projections",  # media_assets
    "knowledge_fact_terms",  # knowledge_fact_entries / knowledge_fact_revisions
    "knowledge_fact_revisions",  # knowledge_fact_entries
)

# “保留表情包重开” retains only the usable sticker inventory and its
# quality metadata.  Candidate/task/history tables are deliberately absent.
INSTANCE_RESET_STICKER_INVENTORY_TABLES = (
    "sticker_reinforcements",
    "sticker_fingerprints",
    "sticker_items",
    "sticker_clusters",
    "sticker_libraries",
)

INSTANCE_RESET_PRESERVED_TABLES = (
    "character_instances",
    "instance_core_state",
    "instance_chat_policies",
    "instance_contact_overrides",
    "instance_delivery_overrides",
    "instance_state_gate_overrides",
    # Cleanup intents must outlive the rows and scopes whose files they own.
    "runtime_file_cleanup_queue",
)


__all__ = [name for name in globals() if name.endswith("_TABLES")]
