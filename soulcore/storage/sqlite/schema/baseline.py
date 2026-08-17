"""The single create-only SQLite schema for SoulCore 1.x."""

from __future__ import annotations

import hashlib

from ....features.stickers.domain import DEFAULT_STICKER_REQUIREMENTS

INSTANCE_CHAT_POLICIES_SQL = r"""
CREATE TABLE instance_chat_policies (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        soulcore_enabled INTEGER NOT NULL DEFAULT 1
            CHECK(soulcore_enabled IN (0, 1)),
        image_send_enabled INTEGER NOT NULL DEFAULT 1
            CHECK(image_send_enabled IN (0, 1)),
        private_fallback_player_name TEXT NOT NULL DEFAULT ''
            CHECK(length(private_fallback_player_name) <= 80),
        private_name_override_enabled INTEGER NOT NULL DEFAULT 0
            CHECK(private_name_override_enabled IN (0, 1)),
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );
""".strip()

SQL = r"""
CREATE TABLE ai_api_models (
        backend_id TEXT PRIMARY KEY
            REFERENCES ai_backends(backend_id) ON DELETE RESTRICT,
        package_id TEXT NOT NULL
            REFERENCES ai_api_packages(package_id) ON DELETE RESTRICT,
        model_key TEXT NOT NULL,
        display_name TEXT NOT NULL DEFAULT '',
        capabilities_json TEXT NOT NULL DEFAULT '[]',
        priority INTEGER NOT NULL CHECK(priority >= 1),
        enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
        archived_at TEXT,
        config_json TEXT NOT NULL DEFAULT '{}',
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

CREATE TABLE ai_api_packages (
        package_id TEXT PRIMARY KEY,
        protocol TEXT NOT NULL DEFAULT 'OPENAI_COMPATIBLE'
            CHECK(protocol IN (
                'OPENAI', 'OPENAI_COMPATIBLE', 'ANTHROPIC',
                'GEMINI', 'CUSTOM_HTTP_IMAGE',
                'MINIMAX_TTS', 'MIMO_TTS', 'GPT_SOVITS_V2', 'GSVI_TTS'
            )),
        display_name TEXT NOT NULL DEFAULT '',
        base_url TEXT NOT NULL DEFAULT '',
        credential_id TEXT NOT NULL DEFAULT '',
        enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
        archived_at TEXT,
        config_json TEXT NOT NULL DEFAULT '{}',
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        profile_id TEXT NOT NULL DEFAULT ''
    );

CREATE TABLE ai_backends (
        backend_id TEXT PRIMARY KEY,
        backend_kind TEXT NOT NULL,
        display_name TEXT NOT NULL DEFAULT '',
        enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
        health_status TEXT NOT NULL DEFAULT 'UNKNOWN',
        circuit_state TEXT NOT NULL DEFAULT 'CLOSED'
            CHECK(circuit_state IN ('CLOSED', 'OPEN', 'HALF_OPEN')),
        consecutive_failures INTEGER NOT NULL DEFAULT 0,
        total_successes INTEGER NOT NULL DEFAULT 0,
        total_failures INTEGER NOT NULL DEFAULT 0,
        opened_at TEXT,
        next_probe_at TEXT,
        last_success_at TEXT,
        last_failure_at TEXT,
        last_error TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

CREATE TABLE ai_capability_pools (
        capability TEXT NOT NULL,
        backend_id TEXT NOT NULL REFERENCES ai_backends(backend_id) ON DELETE CASCADE,
        priority INTEGER NOT NULL DEFAULT 0,
        enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
        config_json TEXT NOT NULL DEFAULT '{}',
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(capability, backend_id)
    );

CREATE TABLE ai_circuit_states (
        circuit_scope TEXT PRIMARY KEY,
        backend_id TEXT NOT NULL REFERENCES ai_backends(backend_id) ON DELETE CASCADE,
        adapter_id TEXT NOT NULL,
        credential_id TEXT NOT NULL DEFAULT '',
        capability TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN (
            'HEALTHY', 'DEGRADED', 'OPEN', 'HALF_OPEN', 'DISABLED'
        )),
        failure_count INTEGER NOT NULL DEFAULT 0 CHECK(failure_count >= 0),
        opened_until TEXT,
        last_error_code TEXT NOT NULL DEFAULT '',
        last_success_at TEXT,
        last_failure_at TEXT,
        updated_at TEXT NOT NULL
    );

CREATE TABLE ai_manager_pauses (
        pause_scope TEXT NOT NULL CHECK(pause_scope IN (
            'GLOBAL', 'BACKGROUND', 'BACKEND', 'CAPABILITY'
        )),
        scope_key TEXT NOT NULL DEFAULT '',
        paused INTEGER NOT NULL DEFAULT 1 CHECK(paused IN (0, 1)),
        reason TEXT NOT NULL DEFAULT '',
        actor_id TEXT NOT NULL DEFAULT '',
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(pause_scope, scope_key)
    );

CREATE TABLE ai_prompt_cache_capabilities (
        backend_id TEXT PRIMARY KEY REFERENCES ai_backends(backend_id) ON DELETE CASCADE,
        model_id TEXT NOT NULL DEFAULT '',
        config_fingerprint TEXT NOT NULL,
        wire_mode TEXT NOT NULL DEFAULT '' CHECK(wire_mode IN (
            '', 'DISABLED', 'OPENAI_AUTO', 'OPENAI_EXPLICIT', 'ANTHROPIC_EPHEMERAL'
        )),
        state TEXT NOT NULL CHECK(state IN (
            'UNTESTED', 'PROBING', 'ACCEPTED_UNVERIFIED', 'CONFIRMED', 'REJECTED'
        )),
        evidence_json TEXT NOT NULL DEFAULT '{}',
        rejected_modes_json TEXT NOT NULL DEFAULT '[]',
        rejection_json TEXT NOT NULL DEFAULT '{}',
        cache_read_tokens INTEGER NOT NULL DEFAULT 0 CHECK(cache_read_tokens >= 0),
        cache_write_tokens INTEGER NOT NULL DEFAULT 0 CHECK(cache_write_tokens >= 0),
        probe_owner TEXT NOT NULL DEFAULT '',
        probe_expires_at TEXT,
        next_probe_at TEXT,
        last_observed_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

CREATE TABLE ai_provider_attempts (
        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        public_ref TEXT NOT NULL UNIQUE,
        node_id INTEGER NOT NULL REFERENCES ai_work_nodes(node_id) ON DELETE CASCADE,
        invocation_id TEXT NOT NULL,
        round_no INTEGER NOT NULL DEFAULT 1 CHECK(round_no >= 1),
        attempt_no INTEGER NOT NULL CHECK(attempt_no >= 1),
        backend_id TEXT REFERENCES ai_backends(backend_id) ON DELETE SET NULL,
        model_id TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL CHECK(status IN (
            'PREPARING', 'IN_FLIGHT', 'SUCCEEDED', 'FAILED',
            'CANCELLED', 'INTERRUPTED'
        )),
        request_json TEXT,
        response_json TEXT,
        transport_json TEXT NOT NULL DEFAULT '{}',
        evaluation_json TEXT,
        error_code TEXT NOT NULL DEFAULT '',
        error_message TEXT NOT NULL DEFAULT '',
        input_tokens INTEGER NOT NULL DEFAULT 0 CHECK(input_tokens >= 0),
        output_tokens INTEGER NOT NULL DEFAULT 0 CHECK(output_tokens >= 0),
        cache_read_tokens INTEGER NOT NULL DEFAULT 0 CHECK(cache_read_tokens >= 0),
        cache_write_tokens INTEGER NOT NULL DEFAULT 0 CHECK(cache_write_tokens >= 0),
        cache_mode TEXT NOT NULL DEFAULT '',
        cache_status TEXT NOT NULL DEFAULT '',
        started_at TEXT NOT NULL,
        sent_at TEXT,
        finished_at TEXT,
        created_at TEXT NOT NULL,
        UNIQUE(node_id, invocation_id, round_no, attempt_no)
    );

CREATE TABLE ai_task_attempts (
        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER NOT NULL REFERENCES "ai_tasks"(task_id) ON DELETE CASCADE,
        attempt_no INTEGER NOT NULL CHECK(attempt_no >= 0),
        lease_token INTEGER NOT NULL CHECK(lease_token >= 0),
        worker_id TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        heartbeat_at TEXT,
        finished_at TEXT,
        error TEXT,
        metrics_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE(task_id, lease_token)
    );

CREATE TABLE ai_task_audit (
        audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER REFERENCES "ai_tasks"(task_id) ON DELETE SET NULL,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        actor_type TEXT NOT NULL,
        actor_id TEXT NOT NULL DEFAULT '',
        action TEXT NOT NULL,
        from_status TEXT,
        to_status TEXT,
        details_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );

CREATE TABLE ai_tasks (
        task_id INTEGER PRIMARY KEY AUTOINCREMENT,
        workflow_id INTEGER REFERENCES ai_workflows(workflow_id) ON DELETE SET NULL,
        caused_by_workflow_id INTEGER REFERENCES ai_workflows(workflow_id) ON DELETE SET NULL,
        origin_work_node_id INTEGER REFERENCES ai_work_nodes(node_id) ON DELETE SET NULL,
        profile_id TEXT NOT NULL REFERENCES role_profiles(profile_id) ON DELETE CASCADE,
        instance_id TEXT NOT NULL,
        task_type TEXT NOT NULL,
        task_class TEXT NOT NULL DEFAULT 'BACKGROUND'
            CHECK(task_class IN ('FOREGROUND', 'BACKGROUND')),
        capability TEXT,
        status TEXT NOT NULL CHECK(status IN (
            'SCHEDULED', 'READY', 'RUNNING',
            'PAUSE_REQUESTED', 'PAUSED', 'CANCEL_REQUESTED', 'RETRY_WAIT',
            'DEFERRED', 'SUCCEEDED', 'FAILED', 'CANCELLED', 'RECOVERY_REQUIRED'
        )),
        priority INTEGER NOT NULL DEFAULT 0,
        due_at TEXT NOT NULL,
        step_key TEXT,
        mutex_key TEXT,
        backend_id TEXT REFERENCES ai_backends(backend_id) ON DELETE SET NULL,
        idempotency_key TEXT,
        generation INTEGER NOT NULL DEFAULT 1 CHECK(generation >= 1),
        input_json TEXT NOT NULL,
        checkpoint_json TEXT NOT NULL,
        result_json TEXT,
        progress_json TEXT NOT NULL DEFAULT '{"schema_version":1,"kind":"progress","payload":{}}',
        retry_policy_json TEXT NOT NULL,
        recovery_policy TEXT NOT NULL DEFAULT 'RESUME_CHECKPOINT'
            CHECK(recovery_policy IN (
                'RESTART_SAFE', 'RESUME_CHECKPOINT',
                'RECONCILE_EXTERNAL', 'NO_RETRY'
            )),
        attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
        max_attempts INTEGER NOT NULL DEFAULT 5 CHECK(max_attempts >= 1),
        lease_owner TEXT,
        lease_token INTEGER NOT NULL DEFAULT 0 CHECK(lease_token >= 0),
        lease_until TEXT,
        last_error TEXT,
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE ai_work_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        public_ref TEXT NOT NULL UNIQUE,
        workflow_id INTEGER NOT NULL REFERENCES ai_workflows(workflow_id) ON DELETE CASCADE,
        node_id INTEGER,
        sequence INTEGER NOT NULL CHECK(sequence >= 1),
        event_category TEXT NOT NULL,
        severity TEXT NOT NULL CHECK(severity IN ('INFO', 'WARNING', 'ERROR')),
        code TEXT NOT NULL DEFAULT '',
        summary TEXT NOT NULL,
        details_json TEXT NOT NULL DEFAULT '{}',
        occurred_at TEXT NOT NULL,
        UNIQUE(workflow_id, sequence),
        FOREIGN KEY(workflow_id, node_id)
            REFERENCES ai_work_nodes(workflow_id, node_id) ON DELETE CASCADE
    );

CREATE TABLE ai_work_nodes (
        node_id INTEGER PRIMARY KEY AUTOINCREMENT,
        public_ref TEXT NOT NULL UNIQUE,
        workflow_id INTEGER NOT NULL REFERENCES ai_workflows(workflow_id) ON DELETE CASCADE,
        parent_node_id INTEGER,
        sequence INTEGER NOT NULL CHECK(sequence >= 1),
        node_role TEXT NOT NULL CHECK(node_role IN (
            'BUSINESS_STAGE', 'INTERNAL_ACTION', 'SYSTEM_STAGE'
        )),
        node_kind TEXT NOT NULL CHECK(node_kind IN (
            'MODEL', 'WEB', 'IMAGE', 'AUDIO', 'FILE', 'COMMAND', 'SYSTEM'
        )),
        purpose TEXT NOT NULL,
        node_key TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'RUNNING' CHECK(status IN (
            'RUNNING', 'SUCCEEDED', 'SKIPPED', 'FALLBACK',
            'FAILED', 'CANCELLED', 'INTERRUPTED'
        )),
        error_code TEXT NOT NULL DEFAULT '',
        error_message TEXT NOT NULL DEFAULT '',
        warning_code TEXT NOT NULL DEFAULT '',
        warning_message TEXT NOT NULL DEFAULT '',
        summary TEXT NOT NULL DEFAULT '',
        input_json TEXT NOT NULL DEFAULT '{}',
        result_json TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        created_at TEXT NOT NULL,
        UNIQUE(workflow_id, sequence),
        UNIQUE(workflow_id, node_key),
        UNIQUE(workflow_id, node_id),
        FOREIGN KEY(workflow_id, parent_node_id)
            REFERENCES ai_work_nodes(workflow_id, node_id) ON DELETE CASCADE
    );

CREATE TABLE ai_workflows (
        workflow_id INTEGER PRIMARY KEY AUTOINCREMENT,
        public_ref TEXT NOT NULL UNIQUE,
        profile_id TEXT NOT NULL REFERENCES role_profiles(profile_id) ON DELETE CASCADE,
        instance_id TEXT,
        workflow_kind TEXT NOT NULL CHECK(workflow_kind IN (
            'CONVERSATION', 'PROACTIVE', 'BACKGROUND', 'ADMIN_TEST'
        )),
        primary_purpose TEXT NOT NULL,
        trigger_kind TEXT NOT NULL,
        trigger_ref TEXT NOT NULL DEFAULT '',
        caused_by_workflow_id INTEGER REFERENCES ai_workflows(workflow_id) ON DELETE SET NULL,
        status TEXT NOT NULL DEFAULT 'RUNNING' CHECK(status IN (
            'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED', 'INTERRUPTED'
        )),
        final_error_code TEXT NOT NULL DEFAULT '',
        final_message TEXT NOT NULL DEFAULT '',
        reason TEXT NOT NULL DEFAULT '',
        idempotency_key TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        expires_at TEXT,
        updated_at TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        UNIQUE(profile_id, idempotency_key),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE background_author_publications (
        publication_id INTEGER PRIMARY KEY AUTOINCREMENT,
        public_ref TEXT NOT NULL UNIQUE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        author_kind TEXT NOT NULL CHECK(author_kind IN (
            'WORLD', 'LIFE_DIRECTION', 'STORY_SOURCE', 'KEYFRAME', 'ORDINARY'
        )),
        generation INTEGER NOT NULL CHECK(generation >= 1),
        state_version INTEGER NOT NULL CHECK(state_version >= 1),
        content_json TEXT NOT NULL DEFAULT '{}',
        outcome_json TEXT NOT NULL DEFAULT '{}',
        creator_output_json TEXT NOT NULL DEFAULT '{}',
        input_versions_json TEXT NOT NULL DEFAULT '{}',
        frame_start_at TEXT,
        frame_end_at TEXT,
        task_id INTEGER REFERENCES ai_tasks(task_id) ON DELETE SET NULL,
        published_at TEXT NOT NULL,
        UNIQUE(profile_id, instance_id, author_kind, generation),
        FOREIGN KEY(profile_id, instance_id, author_kind)
            REFERENCES background_author_states(profile_id, instance_id, author_kind)
            ON DELETE CASCADE,
        CHECK((frame_start_at IS NULL) = (frame_end_at IS NULL)),
        CHECK(frame_end_at IS NULL OR frame_end_at >= frame_start_at)
    );

CREATE TABLE background_author_states (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        author_kind TEXT NOT NULL CHECK(author_kind IN (
            'WORLD', 'LIFE_DIRECTION', 'STORY_SOURCE', 'KEYFRAME', 'ORDINARY'
        )),
        state_version INTEGER NOT NULL DEFAULT 0 CHECK(state_version >= 0),
        schedule_version INTEGER NOT NULL DEFAULT 1 CHECK(schedule_version >= 1),
        state_json TEXT NOT NULL DEFAULT '{}',
        backend_id TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'IDLE'
            CHECK(status IN ('IDLE', 'ENQUEUED', 'RUNNING', 'FAILED')),
        next_due_at TEXT,
        hard_due_at TEXT,
        last_started_at TEXT,
        last_success_at TEXT,
        last_publication_id INTEGER,
        active_task_id INTEGER REFERENCES ai_tasks(task_id) ON DELETE SET NULL,
        generation INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0),
        failure_count INTEGER NOT NULL DEFAULT 0 CHECK(failure_count >= 0),
        last_error TEXT NOT NULL DEFAULT '',
        force_generation INTEGER NOT NULL DEFAULT 0 CHECK(force_generation >= 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id, author_kind),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES background_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE background_initialization_openings (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        anchor_at TEXT NOT NULL,
        keyframe_completed INTEGER NOT NULL DEFAULT 0
            CHECK(keyframe_completed IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES background_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE background_instances (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0, 1)),
        initialization_state TEXT NOT NULL DEFAULT 'UNINITIALIZED'
            CHECK(initialization_state IN ('UNINITIALIZED', 'INITIALIZING', 'READY')),
        initialization_step TEXT NOT NULL DEFAULT 'WORLD'
            CHECK(initialization_step IN (
                'WORLD', 'LIFE_DIRECTION', 'STORY_SOURCE',
                'ORDINARY_CURRENT', 'READY'
            )),
        initial_life_direction TEXT NOT NULL DEFAULT '',
        default_backend_id TEXT NOT NULL DEFAULT '',
        ordinary_min_minutes INTEGER NOT NULL DEFAULT 120
            CHECK(ordinary_min_minutes BETWEEN 1 AND 525600),
        ordinary_max_minutes INTEGER NOT NULL DEFAULT 600
            CHECK(ordinary_max_minutes >= ordinary_min_minutes
                AND ordinary_max_minutes <= 525600),
        keyframe_every_ordinary INTEGER NOT NULL DEFAULT 2
            CHECK(keyframe_every_ordinary BETWEEN 1 AND 100),
        keyframe_max_minutes INTEGER NOT NULL DEFAULT 1440
            CHECK(keyframe_max_minutes BETWEEN 1 AND 525600),
        story_source_min_minutes INTEGER NOT NULL DEFAULT 1440
            CHECK(story_source_min_minutes BETWEEN 1 AND 525600),
        story_source_max_minutes INTEGER NOT NULL DEFAULT 4320
            CHECK(story_source_max_minutes >= story_source_min_minutes
                AND story_source_max_minutes <= 525600),
        life_direction_min_minutes INTEGER NOT NULL DEFAULT 20160
            CHECK(life_direction_min_minutes BETWEEN 1 AND 525600),
        life_direction_max_minutes INTEGER NOT NULL DEFAULT 43200
            CHECK(life_direction_max_minutes >= life_direction_min_minutes
                AND life_direction_max_minutes <= 525600),
        world_min_minutes INTEGER NOT NULL DEFAULT 30240
            CHECK(world_min_minutes BETWEEN 1 AND 525600),
        world_max_minutes INTEGER NOT NULL DEFAULT 50400
            CHECK(world_max_minutes >= world_min_minutes
                AND world_max_minutes <= 525600),
        ordinary_since_keyframe INTEGER NOT NULL DEFAULT 0
            CHECK(ordinary_since_keyframe >= 0),
        continuity_version INTEGER NOT NULL DEFAULT 0
            CHECK(continuity_version >= 0),
        simulated_through_at TEXT,
        foreground_message_cursor INTEGER NOT NULL DEFAULT 0
            CHECK(foreground_message_cursor >= 0),
        foreground_run_cursor INTEGER NOT NULL DEFAULT 0
            CHECK(foreground_run_cursor >= 0),
        last_foreground_at TEXT,
        foreground_lease_owner TEXT,
        foreground_lease_token TEXT,
        foreground_lease_until TEXT,
        foreground_lease_holders_json TEXT NOT NULL DEFAULT '{}'
            CHECK(
                json_valid(foreground_lease_holders_json)
                AND json_type(foreground_lease_holders_json) = 'object'
            ),
        foreground_lease_count INTEGER NOT NULL DEFAULT 0
            CHECK(foreground_lease_count >= 0),
        publication_version INTEGER NOT NULL DEFAULT 0
            CHECK(publication_version >= 0),
        timeline_version INTEGER NOT NULL DEFAULT 0 CHECK(timeline_version >= 0),
        view_version INTEGER NOT NULL DEFAULT 0 CHECK(view_version >= 0),
        config_version INTEGER NOT NULL DEFAULT 1 CHECK(config_version >= 1),
        disabled_at TEXT,
        resumed_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL, proactive_frame_prewarm_enabled INTEGER NOT NULL DEFAULT 1
        CHECK(proactive_frame_prewarm_enabled IN (0, 1)), last_proactive_frame_attempt_at TEXT,
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        CHECK(
            (
                foreground_lease_count = 0
                AND foreground_lease_owner IS NULL
                AND foreground_lease_token IS NULL
                AND foreground_lease_until IS NULL
                AND foreground_lease_holders_json = '{}'
            )
            OR
            (
                foreground_lease_count > 0
                AND foreground_lease_owner IS NOT NULL
                AND foreground_lease_token IS NOT NULL
                AND foreground_lease_until IS NOT NULL
                AND foreground_lease_holders_json <> '{}'
            )
        )
    );

CREATE TABLE background_role_current_views (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
        narrative_time TEXT NOT NULL DEFAULT '',
        location TEXT NOT NULL DEFAULT '',
        doing TEXT NOT NULL DEFAULT '',
        body_state TEXT NOT NULL DEFAULT '',
        mood TEXT NOT NULL DEFAULT '',
        intention TEXT NOT NULL DEFAULT '',
        current_concern TEXT NOT NULL DEFAULT '',
        as_of TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'INITIALIZATION'
            CHECK(source IN ('INITIALIZATION', 'ORDINARY', 'KEYFRAME')),
        source_event_id INTEGER
            CHECK(source_event_id IS NULL OR source_event_id >= 1),
        source_publication_id INTEGER
            REFERENCES background_author_publications(publication_id) ON DELETE SET NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES background_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE background_role_timeline_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        public_ref TEXT NOT NULL UNIQUE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        source TEXT NOT NULL CHECK(source IN ('ORDINARY', 'KEYFRAME')),
        content TEXT NOT NULL,
        frame_start_at TEXT NOT NULL,
        frame_end_at TEXT NOT NULL,
        leftover_text TEXT NOT NULL DEFAULT '',
        source_publication_id INTEGER
            REFERENCES background_author_publications(publication_id) ON DELETE SET NULL,
        created_at TEXT NOT NULL, leftover_retired_at TEXT,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES background_instances(profile_id, instance_id) ON DELETE CASCADE,
        CHECK(frame_end_at >= frame_start_at)
    );

CREATE TABLE background_story_run_exposures (
    run_id INTEGER NOT NULL
        REFERENCES instance_core_runs(run_id) ON DELETE CASCADE,
    story_source_id INTEGER NOT NULL
        REFERENCES background_story_sources(story_source_id) ON DELETE CASCADE,
    first_invocation_id TEXT NOT NULL,
    first_round_no INTEGER NOT NULL CHECK(first_round_no >= 1),
    exposed_at TEXT NOT NULL,
    PRIMARY KEY (run_id, story_source_id)
);

CREATE TABLE background_story_sources (
        story_source_id INTEGER PRIMARY KEY AUTOINCREMENT,
        public_ref TEXT NOT NULL UNIQUE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        module_text TEXT NOT NULL,
        source_publication_id INTEGER
            REFERENCES background_author_publications(publication_id) ON DELETE SET NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL, shown_count INTEGER NOT NULL DEFAULT 0, last_shown_at TEXT, engagement_state TEXT NOT NULL DEFAULT 'PENDING'
    CHECK(engagement_state IN ('PENDING','ACTIVE','CONCLUDED')),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES background_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE background_timeline_event_story_sources (
    event_id INTEGER NOT NULL
        REFERENCES background_role_timeline_events(event_id) ON DELETE CASCADE,
    story_source_id INTEGER NOT NULL
        REFERENCES background_story_sources(story_source_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (event_id, story_source_id)
);

CREATE TABLE character_identity_references (
        reference_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        scope TEXT NOT NULL CHECK(scope IN ('private', 'group')),
        asset_id TEXT NOT NULL UNIQUE,
        storage_relpath TEXT NOT NULL UNIQUE,
        mime_type TEXT NOT NULL,
        file_extension TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        byte_size INTEGER NOT NULL CHECK(byte_size > 0),
        width INTEGER NOT NULL CHECK(width > 0),
        height INTEGER NOT NULL CHECK(height > 0),
        frame_count INTEGER NOT NULL DEFAULT 1 CHECK(frame_count >= 1),
        duration_ms INTEGER NOT NULL DEFAULT 0 CHECK(duration_ms >= 0),
        label TEXT NOT NULL DEFAULT '',
        identity_description TEXT NOT NULL DEFAULT '',
        file_status TEXT NOT NULL DEFAULT 'AVAILABLE' CHECK(file_status IN (
            'AVAILABLE', 'RELEASE_PENDING', 'RELEASED'
        )),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(profile_id, scope),
        FOREIGN KEY(profile_id, scope)
            REFERENCES scope_configs(profile_id, scope) ON DELETE CASCADE
    );

CREATE TABLE character_instances (
        profile_id TEXT NOT NULL REFERENCES role_profiles(profile_id) ON DELETE CASCADE,
        instance_id TEXT NOT NULL,
        route_umo TEXT NOT NULL,
        platform_id TEXT NOT NULL DEFAULT '',
        message_type TEXT NOT NULL DEFAULT '',
        target_id TEXT NOT NULL DEFAULT '',
        scope TEXT NOT NULL CHECK(scope IN ('private', 'group')),
        session_kind TEXT NOT NULL DEFAULT '',
        readiness TEXT NOT NULL DEFAULT 'READY',
        initialization_state TEXT NOT NULL DEFAULT 'READY'
            CHECK(initialization_state IN ('UNINITIALIZED', 'INITIALIZING', 'READY')),
        initialization_completed_at TEXT,
        proactive_enabled INTEGER NOT NULL DEFAULT 1 CHECK(proactive_enabled IN (0, 1)),
        extra_background TEXT NOT NULL DEFAULT '',
        min_wakeup_minutes INTEGER NOT NULL DEFAULT 15 CHECK(min_wakeup_minutes >= 1),
        max_wakeup_minutes INTEGER NOT NULL DEFAULT 55 CHECK(max_wakeup_minutes >= min_wakeup_minutes),
        low_frequency_min_wakeup_minutes INTEGER NOT NULL DEFAULT 180
            CHECK(low_frequency_min_wakeup_minutes >= 1),
        low_frequency_max_wakeup_minutes INTEGER NOT NULL DEFAULT 480
            CHECK(low_frequency_max_wakeup_minutes >= low_frequency_min_wakeup_minutes),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id),
        UNIQUE(profile_id, route_umo)
    );

CREATE TABLE character_intent_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        intent_id TEXT NOT NULL REFERENCES character_intents(intent_id) ON DELETE CASCADE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        event_kind TEXT NOT NULL,
        from_status TEXT,
        to_status TEXT,
        actor_kind TEXT NOT NULL,
        actor_id TEXT NOT NULL DEFAULT '',
        reason TEXT NOT NULL DEFAULT '',
        details_json TEXT NOT NULL DEFAULT '{}',
        source_run_id INTEGER,
        created_at TEXT NOT NULL
    );

CREATE TABLE character_intent_evidence (
        evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
        intent_id TEXT NOT NULL REFERENCES character_intents(intent_id) ON DELETE CASCADE,
        revision INTEGER NOT NULL,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        evidence_kind TEXT NOT NULL CHECK(evidence_kind IN (
            'CURRENT_PLAYER_MESSAGE', 'CORE_RUN', 'ADMIN'
        )),
        source_message_id INTEGER,
        source_run_id INTEGER,
        quote_hash TEXT NOT NULL DEFAULT '',
        quote_offset INTEGER CHECK(quote_offset IS NULL OR quote_offset >= 0),
        quote_length INTEGER CHECK(quote_length IS NULL OR quote_length >= 0),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(intent_id, revision)
            REFERENCES character_intent_revisions(intent_id, revision) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, source_message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id)
            ON DELETE RESTRICT
    );

CREATE TABLE character_intent_revisions (
        intent_id TEXT NOT NULL REFERENCES character_intents(intent_id) ON DELETE CASCADE,
        revision INTEGER NOT NULL CHECK(revision >= 1),
        goal TEXT NOT NULL,
        summary TEXT NOT NULL,
        motivation TEXT NOT NULL DEFAULT '',
        constraints_json TEXT NOT NULL DEFAULT '[]',
        change_reason TEXT NOT NULL DEFAULT '',
        actor_kind TEXT NOT NULL DEFAULT 'MAIN_CORE' CHECK(actor_kind IN (
            'MAIN_CORE', 'BACKGROUND_AUTHOR', 'ADMIN', 'SYSTEM'
        )),
        source_run_id INTEGER,
        created_at TEXT NOT NULL,
        PRIMARY KEY(intent_id, revision)
    );

CREATE TABLE character_intents (
        intent_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        intent_kind TEXT NOT NULL
            CHECK(intent_kind IN ('FUTURE_THOUGHT', 'ACTION_INTENT')),
        origin_kind TEXT NOT NULL CHECK(origin_kind IN (
            'CORE_SELF', 'PLAYER_SUGGESTED', 'PLAYER_REQUESTED', 'ADMIN'
        )),
        status TEXT NOT NULL CHECK(status IN (
            'OPEN', 'CONSUMED', 'PLANNED', 'IN_PROGRESS', 'BLOCKED',
            'COMPLETED', 'CANCELLED', 'EXPIRED', 'SUPERSEDED'
        )),
        current_revision INTEGER NOT NULL DEFAULT 1 CHECK(current_revision >= 1),
        priority REAL NOT NULL DEFAULT 0.5 CHECK(priority >= 0 AND priority <= 1),
        not_before_at TEXT,
        target_at TEXT,
        expires_at TEXT NOT NULL,
        next_review_at TEXT,
        creation_key TEXT NOT NULL,
        content_fingerprint TEXT NOT NULL,
        conflict_key TEXT NOT NULL DEFAULT '',
        source_run_id INTEGER,
        resolution_run_id INTEGER,
        resolved_at TEXT,
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(profile_id, instance_id, creation_key),
        UNIQUE(profile_id, instance_id, intent_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, source_run_id)
            REFERENCES instance_core_runs(profile_id, instance_id, run_id),
        FOREIGN KEY(profile_id, instance_id, resolution_run_id)
            REFERENCES instance_core_runs(profile_id, instance_id, run_id)
    );

CREATE TABLE character_model_revisions (
        profile_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK(revision >= 1),
        schema_version INTEGER NOT NULL CHECK(schema_version = 4),
        content_fingerprint TEXT NOT NULL CHECK(length(content_fingerprint) = 64),
        request_fingerprint TEXT NOT NULL CHECK(length(request_fingerprint) = 64),
        idempotency_key TEXT NOT NULL CHECK(length(idempotency_key) BETWEEN 1 AND 200),
        is_complete INTEGER NOT NULL CHECK(is_complete IN (0, 1)),
        missing_fields_json TEXT NOT NULL,
        identity_json TEXT NOT NULL,
        personality_json TEXT NOT NULL,
        social_json TEXT NOT NULL,
        preferences_json TEXT NOT NULL,
        language_json TEXT NOT NULL,
        dialogue_reference TEXT NOT NULL,
        visual_json TEXT NOT NULL,
        capabilities_json TEXT NOT NULL,
        trigger_rules_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, revision),
        UNIQUE(profile_id, idempotency_key),
        FOREIGN KEY(profile_id) REFERENCES role_profiles(profile_id) ON DELETE CASCADE
    );

CREATE TABLE character_models (
        profile_id TEXT PRIMARY KEY,
        current_revision INTEGER NOT NULL CHECK(current_revision >= 1),
        content_fingerprint TEXT NOT NULL CHECK(length(content_fingerprint) = 64),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(profile_id) REFERENCES role_profiles(profile_id) ON DELETE CASCADE
    );

CREATE TABLE console_preferences (
        preference_key TEXT PRIMARY KEY,
        preference_value TEXT NOT NULL DEFAULT '',
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

CREATE TABLE contact_attempts (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        attempt_ref TEXT NOT NULL,
        generation INTEGER NOT NULL CHECK(generation >= 1),
        task_id INTEGER,
        evidence_snapshot_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'READY' CHECK(status IN ('READY', 'FINALIZED')),
        attempted INTEGER CHECK(attempted IN (0, 1)),
        success INTEGER CHECK(success IN (0, 1)),
        answered INTEGER CHECK(answered IN (0, 1)),
        created_at TEXT NOT NULL,
        finalized_at TEXT, answered_message_id INTEGER, answered_at TEXT,
        PRIMARY KEY(profile_id, instance_id, attempt_ref),
        FOREIGN KEY(profile_id, instance_id) REFERENCES character_instances(profile_id, instance_id)
            ON DELETE CASCADE
    );

CREATE TABLE contact_evidence_reservations (
        reservation_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        attempt_ref TEXT NOT NULL,
        contact_generation INTEGER NOT NULL CHECK(contact_generation >= 1),
        evidence_kind TEXT NOT NULL CHECK(evidence_kind IN (
            'ROLE_TIMELINE_EVENT', 'ACTION_RESULT'
        )),
        evidence_ref TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'RESERVED' CHECK(status IN (
            'RESERVED', 'CONSUMED', 'RELEASED', 'STALE'
        )),
        reserved_at TEXT NOT NULL,
        resolved_at TEXT,
        resolution_reason TEXT NOT NULL DEFAULT '',
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        UNIQUE(profile_id, instance_id, attempt_ref, evidence_kind, evidence_ref),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE contact_expedite_events (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        requested_at TEXT NOT NULL,
        requested_due_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id, event_id),
        FOREIGN KEY(profile_id, instance_id) REFERENCES character_instances(profile_id, instance_id)
            ON DELETE CASCADE
    );

CREATE TABLE contact_policies (
        profile_id TEXT NOT NULL,
        scope TEXT NOT NULL CHECK(scope IN ('private', 'group')),
        proactive_enabled INTEGER NOT NULL DEFAULT 1 CHECK(proactive_enabled IN (0, 1)),
        check_min_minutes INTEGER NOT NULL DEFAULT 180 CHECK(check_min_minutes >= 1),
        check_max_minutes INTEGER NOT NULL DEFAULT 480 CHECK(check_max_minutes >= check_min_minutes),
        quiet_enabled INTEGER NOT NULL DEFAULT 1 CHECK(quiet_enabled IN (0, 1)),
        quiet_start TEXT NOT NULL DEFAULT '23:00',
        quiet_end TEXT NOT NULL DEFAULT '08:00',
        min_success_gap_minutes INTEGER NOT NULL DEFAULT 120 CHECK(min_success_gap_minutes >= 0),
        daily_limit_mode TEXT NOT NULL DEFAULT 'LIMITED'
            CHECK(daily_limit_mode IN ('LIMITED', 'UNLIMITED')),
        daily_success_limit INTEGER DEFAULT 6,
        unanswered_limit_mode TEXT NOT NULL DEFAULT 'UNLIMITED'
            CHECK(unanswered_limit_mode IN ('LIMITED', 'UNLIMITED')),
        max_consecutive_unanswered INTEGER,
        failure_mode TEXT NOT NULL DEFAULT 'SKIP' CHECK(failure_mode IN ('SKIP', 'RETRY_BACKOFF')),
        retry_delay_minutes INTEGER NOT NULL DEFAULT 15 CHECK(retry_delay_minutes >= 1),
        retry_max_attempts INTEGER NOT NULL DEFAULT 3 CHECK(retry_max_attempts >= 0),
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK((daily_limit_mode = 'LIMITED' AND daily_success_limit >= 1)
            OR (daily_limit_mode = 'UNLIMITED' AND daily_success_limit IS NULL)),
        CHECK((unanswered_limit_mode = 'LIMITED' AND max_consecutive_unanswered >= 1)
            OR (unanswered_limit_mode = 'UNLIMITED' AND max_consecutive_unanswered IS NULL)),
        PRIMARY KEY(profile_id, scope),
        FOREIGN KEY(profile_id, scope) REFERENCES scope_configs(profile_id, scope)
            ON DELETE CASCADE
    );

CREATE TABLE context_build_reports (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        model_id TEXT NOT NULL DEFAULT '',
        token_count_mode TEXT NOT NULL DEFAULT 'ESTIMATED',
        hard_token_limit INTEGER NOT NULL,
        target_token_budget INTEGER NOT NULL,
        fill_budget INTEGER NOT NULL,
        total_tokens INTEGER NOT NULL DEFAULT 0,
        report_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE conversation_turn_buffer_batches (
        batch_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        generation INTEGER NOT NULL DEFAULT 1 CHECK(generation >= 1),
        activity_epoch INTEGER NOT NULL CHECK(activity_epoch >= 0),
        status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN (
            'PENDING', 'CLASSIFYING', 'WAITING', 'CLAIMED',
            'RESOLVED', 'CANCELLED', 'FAILED'
        )),
        requested_delay_seconds INTEGER
            CHECK(requested_delay_seconds IS NULL OR
                  requested_delay_seconds BETWEEN 0 AND 60),
        ai_elapsed_seconds REAL
            CHECK(ai_elapsed_seconds IS NULL OR ai_elapsed_seconds >= 0),
        remaining_delay_seconds REAL
            CHECK(remaining_delay_seconds IS NULL OR remaining_delay_seconds >= 0),
        due_at TEXT,
        lease_owner TEXT,
        lease_token INTEGER NOT NULL DEFAULT 0 CHECK(lease_token >= 0),
        lease_until TEXT,
        main_core_task_ref TEXT,
        error_code TEXT NOT NULL DEFAULT '',
        resolution_outcome TEXT NOT NULL DEFAULT '',
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        resolved_at TEXT,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        CHECK((status IN ('WAITING', 'CLAIMED')) = (due_at IS NOT NULL)),
        CHECK((lease_owner IS NULL) = (lease_until IS NULL)),
        CHECK(status IN ('CLASSIFYING', 'CLAIMED') OR lease_owner IS NULL)
    );

CREATE TABLE conversation_turn_buffer_members (
        batch_id TEXT NOT NULL REFERENCES conversation_turn_buffer_batches(batch_id)
            ON DELETE CASCADE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        message_id INTEGER NOT NULL,
        ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
        added_at TEXT NOT NULL,
        PRIMARY KEY(batch_id, message_id),
        UNIQUE(batch_id, ordinal),
        FOREIGN KEY(profile_id, instance_id, message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id)
            ON DELETE CASCADE
    );

CREATE TABLE creative_boundaries (
        boundary_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL
            REFERENCES role_profiles(profile_id) ON DELETE CASCADE,
        revision INTEGER NOT NULL DEFAULT 1 CHECK(revision >= 1),
        severity TEXT NOT NULL CHECK(severity IN ('HARD', 'PREFERENCE')),
        category TEXT NOT NULL DEFAULT 'CUSTOM',
        rule_text TEXT NOT NULL,
        positive_space TEXT NOT NULL DEFAULT '',
        enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

CREATE TABLE deferred_message_batches (
        batch_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN (
            'PENDING', 'CLAIMED', 'MERGED', 'RESOLVED', 'CANCELLED',
            'EXPIRED', 'FAILED'
        )),
        due_at TEXT NOT NULL,
        activity_epoch INTEGER NOT NULL CHECK(activity_epoch >= 0),
        gate_generation INTEGER NOT NULL CHECK(gate_generation >= 1),
        creation_key TEXT NOT NULL,
        lease_until TEXT,
        lease_token INTEGER NOT NULL DEFAULT 0 CHECK(lease_token >= 0),
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        resolution_reason TEXT NOT NULL DEFAULT '',
        resolution_run_id INTEGER,
        resolved_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(profile_id, instance_id, creation_key),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, resolution_run_id)
            REFERENCES instance_core_runs(profile_id, instance_id, run_id)
            DEFERRABLE INITIALLY DEFERRED
    );

CREATE TABLE deferred_message_items (
        batch_id TEXT NOT NULL REFERENCES deferred_message_batches(batch_id)
            ON DELETE CASCADE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        message_id INTEGER NOT NULL,
        message_ref TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        activity_epoch INTEGER NOT NULL CHECK(activity_epoch >= 0),
        received_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PENDING'
            CHECK(status IN ('PENDING', 'MERGED', 'RESOLVED', 'CANCELLED')),
        added_at TEXT NOT NULL,
        resolved_at TEXT,
        PRIMARY KEY(batch_id, message_id),
        UNIQUE(profile_id, instance_id, message_id),
        UNIQUE(profile_id, instance_id, idempotency_key),
        FOREIGN KEY(profile_id, instance_id, message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id)
            ON DELETE CASCADE
    );

CREATE TABLE dialogue_summaries (
        summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        version INTEGER NOT NULL CHECK(version >= 1),
        strategy_id TEXT NOT NULL,
        strategy_version INTEGER NOT NULL CHECK(strategy_version >= 1),
        covered_from_message_id INTEGER NOT NULL,
        covered_through_message_id INTEGER NOT NULL,
        structured_json TEXT NOT NULL DEFAULT '{}',
        rendered_text TEXT NOT NULL DEFAULT '',
        token_count INTEGER NOT NULL DEFAULT 0 CHECK(token_count >= 0),
        created_at TEXT NOT NULL,
        UNIQUE(profile_id, instance_id, version),
        UNIQUE(profile_id, instance_id, summary_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, covered_from_message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id),
        FOREIGN KEY(profile_id, instance_id, covered_through_message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id)
    );

CREATE TABLE expression_interruption_events (
        interruption_id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL REFERENCES instance_expression_batches(batch_id)
            ON DELETE CASCADE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        inbound_message_id INTEGER NOT NULL,
        context_note TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE(batch_id, inbound_message_id),
        FOREIGN KEY(profile_id, instance_id, inbound_message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id)
            ON DELETE CASCADE
    );

CREATE TABLE file_assets (
        asset_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        job_id TEXT UNIQUE,
        file_format TEXT NOT NULL CHECK(file_format IN ('MD', 'TXT', 'PDF')),
        display_name TEXT NOT NULL,
        mime_type TEXT NOT NULL,
        storage_relpath TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        byte_size INTEGER NOT NULL CHECK(byte_size > 0),
        char_count INTEGER NOT NULL DEFAULT 0 CHECK(char_count >= 0),
        page_count INTEGER NOT NULL DEFAULT 0 CHECK(page_count >= 0),
        file_status TEXT NOT NULL DEFAULT 'AVAILABLE' CHECK(file_status IN (
            'AVAILABLE', 'MISSING', 'QUARANTINED', 'RELEASE_PENDING', 'RELEASED'
        )),
        delivery_status TEXT NOT NULL DEFAULT 'NOT_SELECTED' CHECK(delivery_status IN (
            'NOT_SELECTED', 'SELECTED', 'OUTBOX_PENDING',
            'PLATFORM_ACCEPTED_UNCONFIRMED', 'UNKNOWN_AFTER_CRASH',
            'DELIVERED', 'FAILED'
        )),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        expires_at TEXT NOT NULL,
        released_at TEXT,
        last_error TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE file_generation_jobs (
        job_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        source_run_id INTEGER NOT NULL,
        ai_task_id INTEGER NOT NULL UNIQUE
            REFERENCES ai_tasks(task_id) ON DELETE RESTRICT,
        file_format TEXT NOT NULL CHECK(file_format IN ('MD', 'TXT', 'PDF')),
        display_name TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'QUEUED' CHECK(status IN (
            'QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED',
            'RECOVERY_REQUIRED'
        )),
        safe_error_code TEXT NOT NULL DEFAULT '',
        safe_error_message TEXT NOT NULL DEFAULT '',
        idempotency_key TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        finished_at TEXT,
        UNIQUE(profile_id, instance_id, idempotency_key),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, source_run_id)
            REFERENCES instance_core_runs(profile_id, instance_id, run_id)
            ON DELETE RESTRICT
    );

CREATE TABLE group_flow_instance_state (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        rate_ewma REAL NOT NULL DEFAULT 0 CHECK(rate_ewma >= 0),
        last_inbound_at TEXT,
        last_visible_assistant_at TEXT,
        last_judged_message_id INTEGER,
        last_resolved_message_id INTEGER,
        activity_released_through_message_id INTEGER,
        algorithm_version INTEGER NOT NULL DEFAULT 1 CHECK(algorithm_version >= 1),
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, last_judged_message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id),
        FOREIGN KEY(profile_id, instance_id, last_resolved_message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id),
        FOREIGN KEY(profile_id, instance_id, activity_released_through_message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id)
    );

CREATE TABLE group_flow_policies (
        profile_id TEXT NOT NULL REFERENCES role_profiles(profile_id) ON DELETE CASCADE,
        scope TEXT NOT NULL DEFAULT 'group' CHECK(scope = 'group'),
        quiet_seconds INTEGER NOT NULL DEFAULT 30 CHECK(quiet_seconds BETWEEN 5 AND 300),
        base_message_count INTEGER NOT NULL DEFAULT 2
            CHECK(base_message_count BETWEEN 1 AND 50),
        ordinary_min_reply_gap_seconds INTEGER NOT NULL DEFAULT 0
            CHECK(ordinary_min_reply_gap_seconds BETWEEN 0 AND 86400),
        judge_token_budget INTEGER NOT NULL DEFAULT 2048
            CHECK(judge_token_budget BETWEEN 512 AND 8192),
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, scope)
    );

CREATE TABLE group_flow_window_members (
        window_id TEXT NOT NULL REFERENCES group_flow_windows(window_id) ON DELETE CASCADE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        message_id INTEGER NOT NULL,
        ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
        normalized_fingerprint TEXT NOT NULL DEFAULT '',
        media_cluster_keys_json TEXT NOT NULL DEFAULT '[]',
        sender_id TEXT NOT NULL DEFAULT '',
        occurred_at TEXT NOT NULL,
        added_at TEXT NOT NULL,
        PRIMARY KEY(window_id, message_id),
        UNIQUE(window_id, ordinal),
        UNIQUE(profile_id, instance_id, message_id),
        FOREIGN KEY(profile_id, instance_id, message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id) ON DELETE CASCADE
    );

CREATE TABLE group_flow_windows (
        window_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'COLLECTING' CHECK(status IN (
            'COLLECTING', 'JUDGING', 'READY', 'RUNNING',
            'WAITING_FIRST_ATTEMPT', 'RESOLVED', 'FAILED', 'CANCELLED'
        )),
        first_message_id INTEGER NOT NULL,
        last_message_id INTEGER NOT NULL,
        message_count INTEGER NOT NULL DEFAULT 1 CHECK(message_count >= 1),
        rate_ewma REAL NOT NULL DEFAULT 0 CHECK(rate_ewma >= 0),
        repeat_ratio REAL NOT NULL DEFAULT 0 CHECK(repeat_ratio BETWEEN 0.0 AND 1.0),
        judge_threshold INTEGER NOT NULL DEFAULT 2
            CHECK(judge_threshold BETWEEN 1 AND 4096),
        judge_through_message_id INTEGER,
        frozen_through_message_id INTEGER,
        next_judge_at TEXT,
        quiet_due_at TEXT,
        dynamic_due_at TEXT,
        direct_due_at TEXT,
        direct_address INTEGER NOT NULL DEFAULT 0 CHECK(direct_address IN (0, 1)),
        judge_result TEXT NOT NULL DEFAULT ''
            CHECK(judge_result IN ('', 'SUITABLE', 'UNSUITABLE')),
        judge_error_code TEXT NOT NULL DEFAULT '',
        ready_at TEXT,
        lease_owner TEXT,
        lease_token INTEGER NOT NULL DEFAULT 0 CHECK(lease_token >= 0),
        lease_until TEXT,
        main_core_task_ref TEXT,
        first_attempt_started_at TEXT,
        error_code TEXT NOT NULL DEFAULT '',
        resolution_outcome TEXT NOT NULL DEFAULT '',
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        resolved_at TEXT,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, first_message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id),
        FOREIGN KEY(profile_id, instance_id, last_message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id),
        FOREIGN KEY(profile_id, instance_id, judge_through_message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id),
        FOREIGN KEY(profile_id, instance_id, frozen_through_message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id),
        CHECK(first_message_id <= last_message_id),
        CHECK((lease_owner IS NULL) = (lease_until IS NULL)),
        CHECK(status IN ('JUDGING', 'RUNNING') OR lease_owner IS NULL)
    );

CREATE TABLE group_reply_relocation_states (
        window_id TEXT PRIMARY KEY
            REFERENCES group_flow_windows(window_id) ON DELETE CASCADE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        relocation_count INTEGER NOT NULL DEFAULT 0
            CHECK(relocation_count BETWEEN 0 AND 1),
        last_checked_message_id INTEGER,
        candidate_through_message_id INTEGER,
        candidate_recheck_at TEXT,
        check_owner TEXT,
        check_token INTEGER NOT NULL DEFAULT 0 CHECK(check_token >= 0),
        check_until TEXT,
        check_through_message_id INTEGER,
        check_final INTEGER NOT NULL DEFAULT 0 CHECK(check_final IN (0, 1)),
        last_error_code TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(profile_id, instance_id, window_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        CHECK((candidate_through_message_id IS NULL) =
              (candidate_recheck_at IS NULL)),
        CHECK((check_owner IS NULL) = (check_until IS NULL)),
        CHECK((check_owner IS NULL) = (check_through_message_id IS NULL)),
        CHECK(check_owner IS NOT NULL OR check_final = 0)
    );

CREATE TABLE important_todos (
        todo_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('FILE_READY', 'FILE_FAILED')),
        source_job_id TEXT NOT NULL UNIQUE
            REFERENCES file_generation_jobs(job_id) ON DELETE RESTRICT,
        file_asset_id TEXT REFERENCES file_assets(asset_id) ON DELETE RESTRICT,
        payload_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN (
            'PENDING', 'SELECTED', 'DELIVERY_PENDING', 'DELIVERY_UNKNOWN',
            'COMPLETED', 'CANCELLED'
        )),
        available_at TEXT NOT NULL,
        selected_run_id INTEGER,
        selected_activity_epoch INTEGER,
        delivery_outbox_id INTEGER REFERENCES instance_outbox(outbox_id)
            ON DELETE SET NULL,
        idempotency_key TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        resolved_at TEXT,
        UNIQUE(profile_id, instance_id, idempotency_key),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, selected_run_id)
            REFERENCES instance_core_runs(profile_id, instance_id, run_id)
            ON DELETE RESTRICT
    );

CREATE TABLE inbound_message_recall_states (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        ledger_message_id INTEGER NOT NULL,
        platform_instance_id TEXT NOT NULL CHECK(length(trim(platform_instance_id)) > 0),
        route_umo TEXT NOT NULL CHECK(length(trim(route_umo)) > 0),
        platform_message_id TEXT NOT NULL CHECK(length(trim(platform_message_id)) > 0),
        scope TEXT NOT NULL CHECK(scope IN ('private', 'group', 'guild')),
        direct_address INTEGER NOT NULL DEFAULT 0 CHECK(direct_address IN (0, 1)),
        received_at TEXT NOT NULL,
        grace_until TEXT NOT NULL,
        previous_activity_at TEXT,
        status TEXT NOT NULL DEFAULT 'HELD' CHECK(status IN (
            'HELD', 'CLAIMED', 'RELEASED', 'DISPATCHED', 'RECALLED', 'FAILED'
        )),
        lease_owner TEXT,
        lease_token INTEGER NOT NULL DEFAULT 0 CHECK(lease_token >= 0),
        lease_until TEXT,
        activity_epoch INTEGER CHECK(activity_epoch IS NULL OR activity_epoch >= 0),
        dispatched_at TEXT,
        committed_full_at TEXT,
        committed_run_id INTEGER,
        recall_received_at TEXT,
        recall_platform_at TEXT,
        recall_sender_id TEXT NOT NULL DEFAULT '',
        recall_operator_id TEXT NOT NULL DEFAULT '',
        algorithm_version INTEGER NOT NULL DEFAULT 1 CHECK(algorithm_version >= 1),
        probability_seen REAL CHECK(
            probability_seen IS NULL OR probability_seen BETWEEN 0.0 AND 1.0
        ),
        attention_sample REAL CHECK(
            attention_sample IS NULL OR attention_sample BETWEEN 0.0 AND 1.0
        ),
        read_sample REAL CHECK(read_sample IS NULL OR read_sample BETWEEN 0.0 AND 1.0),
        read_fraction REAL CHECK(read_fraction IS NULL OR read_fraction BETWEEN 0.0 AND 1.0),
        visibility TEXT CHECK(visibility IS NULL OR visibility IN ('NONE', 'PREFIX', 'FULL')),
        exposed_text TEXT NOT NULL DEFAULT '',
        original_plain_text TEXT NOT NULL DEFAULT '',
        original_components_json TEXT NOT NULL DEFAULT '[]',
        recall_event_message_id INTEGER,
        last_error TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id, ledger_message_id),
        UNIQUE(profile_id, instance_id, platform_instance_id, route_umo, platform_message_id),
        FOREIGN KEY(profile_id, instance_id, ledger_message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, recall_event_message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id) ON DELETE SET NULL,
        CHECK((lease_owner IS NULL) = (lease_until IS NULL)),
        CHECK(status = 'CLAIMED' OR lease_owner IS NULL)
    );

CREATE TABLE inbound_recall_receipts (
        receipt_id TEXT PRIMARY KEY CHECK(length(trim(receipt_id)) > 0),
        profile_id TEXT NOT NULL REFERENCES role_profiles(profile_id) ON DELETE CASCADE,
        instance_id TEXT NOT NULL,
        platform_instance_id TEXT NOT NULL CHECK(length(trim(platform_instance_id)) > 0),
        route_umo TEXT NOT NULL CHECK(length(trim(route_umo)) > 0),
        platform_message_id TEXT NOT NULL CHECK(length(trim(platform_message_id)) > 0),
        notice_type TEXT NOT NULL CHECK(notice_type IN ('friend_recall', 'group_recall')),
        sender_id TEXT NOT NULL DEFAULT '',
        operator_id TEXT NOT NULL DEFAULT '',
        received_at TEXT NOT NULL,
        platform_occurred_at TEXT,
        status TEXT NOT NULL DEFAULT 'UNMATCHED' CHECK(status IN (
            'UNMATCHED', 'PROCESSING', 'RESOLVED', 'IGNORED'
        )),
        matched_ledger_message_id INTEGER,
        expires_at TEXT NOT NULL,
        completed_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(profile_id, instance_id, platform_instance_id, route_umo, platform_message_id)
    );

CREATE TABLE inbound_voice_admissions (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        platform_message_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('PENDING', 'SETTLED')),
        voice_count INTEGER NOT NULL CHECK(voice_count > 0),
        transcripts_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id, platform_message_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

__INSTANCE_CHAT_POLICIES_SQL__

CREATE TABLE instance_contact_overrides (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        proactive_enabled INTEGER CHECK(proactive_enabled IN (0, 1)),
        check_min_minutes INTEGER CHECK(check_min_minutes IS NULL OR check_min_minutes >= 1),
        check_max_minutes INTEGER CHECK(check_max_minutes IS NULL OR check_max_minutes >= 1),
        quiet_enabled INTEGER CHECK(quiet_enabled IN (0, 1)),
        quiet_start TEXT,
        quiet_end TEXT,
        min_success_gap_minutes INTEGER CHECK(min_success_gap_minutes IS NULL OR min_success_gap_minutes >= 0),
        daily_limit_mode TEXT NOT NULL DEFAULT 'INHERIT'
            CHECK(daily_limit_mode IN ('INHERIT', 'LIMITED', 'UNLIMITED')),
        daily_success_limit INTEGER,
        unanswered_limit_mode TEXT NOT NULL DEFAULT 'INHERIT'
            CHECK(unanswered_limit_mode IN ('INHERIT', 'LIMITED', 'UNLIMITED')),
        max_consecutive_unanswered INTEGER,
        failure_mode TEXT CHECK(failure_mode IS NULL OR failure_mode IN ('SKIP', 'RETRY_BACKOFF')),
        retry_delay_minutes INTEGER CHECK(retry_delay_minutes IS NULL OR retry_delay_minutes >= 1),
        retry_max_attempts INTEGER CHECK(retry_max_attempts IS NULL OR retry_max_attempts >= 0),
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK((daily_limit_mode = 'LIMITED' AND daily_success_limit >= 1)
            OR (daily_limit_mode IN ('INHERIT', 'UNLIMITED') AND daily_success_limit IS NULL)),
        CHECK((unanswered_limit_mode = 'LIMITED' AND max_consecutive_unanswered >= 1)
            OR (unanswered_limit_mode IN ('INHERIT', 'UNLIMITED') AND max_consecutive_unanswered IS NULL)),
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id) REFERENCES character_instances(profile_id, instance_id)
            ON DELETE CASCADE
    );

CREATE TABLE instance_contact_state (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        next_check_at TEXT,
        last_check_at TEXT,
        last_attempt_at TEXT,
        last_success_at TEXT,
        daily_bucket TEXT NOT NULL DEFAULT '',
        daily_success_count INTEGER NOT NULL DEFAULT 0 CHECK(daily_success_count >= 0),
        consecutive_unanswered INTEGER NOT NULL DEFAULT 0 CHECK(consecutive_unanswered >= 0),
        cooldown_until TEXT,
        timeline_event_watermark INTEGER NOT NULL DEFAULT 0
            CHECK(timeline_event_watermark >= 0),
        evidence_watermarks_json TEXT NOT NULL DEFAULT '{}',
        deferred_evidence_json TEXT NOT NULL DEFAULT '{}',
        last_result TEXT NOT NULL DEFAULT '',
        last_reason TEXT NOT NULL DEFAULT '',
        last_committed_task_id INTEGER,
        generation INTEGER NOT NULL DEFAULT 1 CHECK(generation >= 1),
        activity_epoch_snapshot INTEGER,
        evidence_snapshot_json TEXT NOT NULL DEFAULT '{}',
        lease_until TEXT,
        lease_token INTEGER NOT NULL DEFAULT 0 CHECK(lease_token >= 0),
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id) REFERENCES character_instances(profile_id, instance_id)
            ON DELETE CASCADE
    );

CREATE TABLE instance_core_runs (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        workflow_id INTEGER UNIQUE REFERENCES ai_workflows(workflow_id) ON DELETE SET NULL,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        source TEXT NOT NULL,
        status TEXT NOT NULL,
        reason TEXT NOT NULL DEFAULT '',
        request_json TEXT NOT NULL DEFAULT '{}',
        decision_json TEXT,
        expected_state_epoch INTEGER,
        committed_state_epoch INTEGER,
        error TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE instance_core_state (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        state_epoch INTEGER NOT NULL DEFAULT 0,
        activity_epoch INTEGER NOT NULL DEFAULT 0,
        low_frequency_mode INTEGER NOT NULL DEFAULT 0 CHECK(low_frequency_mode IN (0, 1)),
        low_frequency_reason TEXT NOT NULL DEFAULT '',
        low_frequency_since TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE instance_delivery_overrides (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        send_qpm_limit INTEGER CHECK(send_qpm_limit IS NULL OR send_qpm_limit >= 1),
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        CHECK(send_qpm_limit IS NOT NULL)
    );

CREATE TABLE instance_delivery_state (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        inbound_message_id TEXT,
        inbound_received_at TEXT,
        passive_reply_uses INTEGER NOT NULL DEFAULT 0,
        wakeup_periods_json TEXT NOT NULL DEFAULT '{}',
        last_mode TEXT,
        last_status TEXT,
        last_error TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE instance_expression_batches (
        batch_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        source_run_id INTEGER NOT NULL,
        segment_index INTEGER NOT NULL DEFAULT 0 CHECK(segment_index >= 0),
        activity_epoch INTEGER NOT NULL CHECK(activity_epoch >= 0),
        route_umo TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN (
            'ACTIVE', 'SETTLED', 'PARTIALLY_SETTLED', 'CANCELLED', 'FAILED'
        )),
        output_count INTEGER NOT NULL CHECK(output_count >= 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        settled_at TEXT,
        retraction_count INTEGER NOT NULL DEFAULT 0
            CHECK(retraction_count BETWEEN 0 AND 6),
        step_count INTEGER GENERATED ALWAYS AS
            (output_count + retraction_count) VIRTUAL,
        UNIQUE(profile_id, instance_id, batch_id),
        UNIQUE(profile_id, instance_id, source_run_id, segment_index),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, source_run_id)
            REFERENCES instance_core_runs(profile_id, instance_id, run_id)
    );

CREATE TABLE instance_main_core_occupancies (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        occupancy_id TEXT NOT NULL CHECK(length(occupancy_id) BETWEEN 1 AND 160),
        kind TEXT NOT NULL CHECK(kind IN ('PLAYER', 'TIMER', 'EXPRESSION')),
        resource_ref TEXT NOT NULL CHECK(length(resource_ref) BETWEEN 1 AND 160),
        status TEXT NOT NULL CHECK(status IN ('ACTIVE', 'RELEASED', 'EXPIRED')),
        version INTEGER NOT NULL CHECK(version >= 1),
        generation INTEGER NOT NULL CHECK(generation >= 0),
        lease_owner TEXT NOT NULL CHECK(length(lease_owner) BETWEEN 1 AND 160),
        lease_token TEXT NOT NULL CHECK(length(lease_token) BETWEEN 1 AND 160),
        lease_expires_at TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        released_at TEXT,
        PRIMARY KEY(profile_id, instance_id, occupancy_id),
        CHECK((status = 'ACTIVE') = (released_at IS NULL)),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE instance_message_fragments (
        message_ref TEXT PRIMARY KEY CHECK(length(trim(message_ref)) > 0),
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        ledger_message_id INTEGER NOT NULL,
        fragment_ordinal INTEGER NOT NULL CHECK(fragment_ordinal >= 0),
        platform_instance_id TEXT NOT NULL
            CHECK(length(trim(platform_instance_id)) > 0),
        route_umo TEXT NOT NULL CHECK(length(trim(route_umo)) > 0),
        platform_message_id TEXT NOT NULL
            CHECK(length(trim(platform_message_id)) > 0),
        direction TEXT NOT NULL CHECK(direction IN ('INBOUND', 'OUTBOUND')),
        content_kind TEXT NOT NULL CHECK(content_kind IN (
            'TEXT', 'IMAGE', 'STICKER', 'FILE', 'OTHER'
        )),
        content_projection TEXT NOT NULL DEFAULT '',
        sender_id TEXT NOT NULL DEFAULT '',
        native_reply_supported INTEGER NOT NULL DEFAULT 0
            CHECK(native_reply_supported IN (0, 1)),
        member_mention_supported INTEGER NOT NULL DEFAULT 0
            CHECK(member_mention_supported IN (0, 1)),
        self_retraction_supported INTEGER NOT NULL DEFAULT 0
            CHECK(self_retraction_supported IN (0, 1)),
        returns_platform_message_id INTEGER NOT NULL DEFAULT 1
            CHECK(returns_platform_message_id IN (0, 1)),
        accepted_at TEXT NOT NULL,
        retractable_until TEXT,
        retraction_status TEXT CHECK(retraction_status IS NULL OR retraction_status IN (
            'PENDING', 'SENDING', 'RETRACTED', 'FAILED',
            'UNKNOWN_AFTER_CRASH', 'CANCELLED'
        )),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL, platform_reference_id TEXT
        NOT NULL DEFAULT '',
        UNIQUE(profile_id, instance_id, ledger_message_id, fragment_ordinal),
        UNIQUE(profile_id, instance_id, message_ref),
        UNIQUE(platform_instance_id, route_umo, platform_message_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, ledger_message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id) ON DELETE CASCADE
    );

CREATE TABLE instance_messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        direction TEXT NOT NULL CHECK(direction IN ('INBOUND', 'OUTBOUND')),
        role TEXT NOT NULL,
        internal_memo TEXT NOT NULL DEFAULT '',
        sender_id TEXT NOT NULL DEFAULT '',
        sender_name TEXT NOT NULL DEFAULT '',
        plain_text TEXT NOT NULL DEFAULT '',
        identity_template TEXT NOT NULL DEFAULT '',
        components_json TEXT NOT NULL DEFAULT '[]',
        delivery_status TEXT NOT NULL,
        idempotency_key TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        occurred_at TEXT NOT NULL,
        created_at TEXT NOT NULL,
        knowledge_eligibility TEXT NOT NULL DEFAULT 'ELIGIBLE'
            CHECK(knowledge_eligibility IN ('ELIGIBLE', 'HELD', 'EXCLUDED')),
        knowledge_eligibility_reason TEXT NOT NULL DEFAULT '',
        expression_batch_id TEXT,
        expression_ordinal INTEGER
            CHECK(expression_ordinal IS NULL OR expression_ordinal >= 0),
        UNIQUE(profile_id, instance_id, message_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE instance_outbox (
        outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
        workflow_id INTEGER REFERENCES ai_workflows(workflow_id) ON DELETE SET NULL,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        route_umo TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PENDING',
        idempotency_key TEXT NOT NULL,
        activity_epoch INTEGER NOT NULL DEFAULT 0,
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error_code TEXT NOT NULL DEFAULT '',
        last_error TEXT,
        last_diagnostic_code TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        origin_kind TEXT NOT NULL,
        origin_run_id INTEGER,
        origin_task_id INTEGER,
        origin_wakeup_id INTEGER,
        origin_generation INTEGER,
        expression_batch_id TEXT REFERENCES instance_expression_batches(batch_id),
        expression_ordinal INTEGER
            CHECK(expression_ordinal IS NULL OR expression_ordinal >= 0),
        not_before_at TEXT,
        interrupt_policy TEXT NOT NULL DEFAULT 'PRESERVE'
            CHECK(interrupt_policy IN ('PRESERVE', 'CANCEL_ON_PLAYER_MESSAGE')),
        depends_on_idempotency_key TEXT,
        context_message_id INTEGER REFERENCES instance_messages(message_id),
        expression_step_ordinal INTEGER
            CHECK(expression_step_ordinal IS NULL OR expression_step_ordinal >= 0),
        UNIQUE(profile_id, instance_id, idempotency_key),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE instance_participant_identities (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        participant_id TEXT NOT NULL,
        display_name TEXT NOT NULL DEFAULT '',
        name_source TEXT NOT NULL DEFAULT 'OBSERVED'
            CHECK(name_source IN ('OBSERVED', 'PLATFORM_REFRESH')),
        last_message_id INTEGER,
        observed_at TEXT NOT NULL,
        refreshed_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id, participant_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE instance_state_gate_overrides (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        enabled INTEGER CHECK(enabled IN (0, 1)),
        silent_enabled INTEGER CHECK(silent_enabled IN (0, 1)),
        max_gate_hours INTEGER CHECK(max_gate_hours IS NULL OR
            (max_gate_hours >= 1 AND max_gate_hours <= 24)),
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE instance_state_gate_snapshots (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        action TEXT NOT NULL DEFAULT 'OPEN'
            CHECK(action IN ('OPEN', 'DECLINE', 'SILENT', 'DEFER')),
        reason_code TEXT NOT NULL DEFAULT '',
        expression_context TEXT NOT NULL DEFAULT '',
        not_before_at TEXT,
        until_at TEXT,
        source_run_id INTEGER,
        generation INTEGER NOT NULL DEFAULT 1 CHECK(generation >= 1),
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK(action = 'OPEN' OR until_at IS NOT NULL),
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, source_run_id)
            REFERENCES instance_core_runs(profile_id, instance_id, run_id)
            DEFERRABLE INITIALLY DEFERRED
    );

CREATE TABLE instance_wakeups (
        wakeup_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        source TEXT NOT NULL CHECK(source IN (
            'FOREGROUND_MESSAGE', 'BACKGROUND_AUTHOR', 'SELF_WAKEUP',
            'PLUGIN_WAKE', 'DEFERRED_MESSAGE', 'TIMER'
        )),
        due_at TEXT NOT NULL,
        reason TEXT NOT NULL DEFAULT '',
        conversation_ref TEXT,
        idempotency_key TEXT,
        payload_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN (
            'PENDING', 'CLAIMED', 'COMPLETED', 'FAILED', 'CANCELLED'
        )),
        attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
        lease_until TEXT,
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        generation INTEGER NOT NULL DEFAULT 1 CHECK(generation >= 1),
        lease_token INTEGER NOT NULL DEFAULT 0 CHECK(lease_token >= 0),
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        intent_kind TEXT NOT NULL CHECK(intent_kind IN (
            'BACKGROUND_AUTHOR', 'SELF_WAKEUP', 'PLUGIN_WAKE'
        )),
        linked_task_id INTEGER,
        handoff_at TEXT,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE knowledge_audit (
        audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        entity_type TEXT NOT NULL CHECK(entity_type IN ('MEMORY', 'KNOWLEDGE_FACT', 'BATCH')),
        entity_id INTEGER,
        action TEXT NOT NULL,
        actor_type TEXT NOT NULL DEFAULT 'SYSTEM',
        actor_id TEXT NOT NULL DEFAULT '',
        reason TEXT NOT NULL DEFAULT '',
        details_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE knowledge_batch_messages (
        batch_id INTEGER NOT NULL REFERENCES knowledge_batches(batch_id) ON DELETE CASCADE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        message_id INTEGER NOT NULL,
        is_boundary INTEGER NOT NULL DEFAULT 0 CHECK(is_boundary IN (0, 1)),
        projected_text TEXT NOT NULL DEFAULT '',
        projection_truncated INTEGER NOT NULL DEFAULT 0
            CHECK(projection_truncated IN (0, 1)),
        PRIMARY KEY(batch_id, message_id),
        FOREIGN KEY(profile_id, instance_id, message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id) ON DELETE CASCADE
    );

CREATE TABLE knowledge_batches (
        batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        ai_task_id INTEGER REFERENCES "ai_tasks"(task_id) ON DELETE SET NULL,
        processing_version INTEGER NOT NULL,
        status TEXT NOT NULL CHECK(status IN (
            'PREPARED', 'COMMITTED', 'FAILED', 'SUPERSEDED'
        )),
        first_message_id INTEGER,
        last_message_id INTEGER,
        message_count INTEGER NOT NULL DEFAULT 0 CHECK(message_count >= 0),
        estimated_tokens INTEGER NOT NULL DEFAULT 0 CHECK(estimated_tokens >= 0),
        boundary_message_ids_json TEXT NOT NULL DEFAULT '[]',
        output_json TEXT,
        rejection_json TEXT NOT NULL DEFAULT '[]',
        error TEXT,
        created_at TEXT NOT NULL,
        committed_at TEXT,
        UNIQUE(profile_id, instance_id, batch_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE knowledge_fact_entries (
        knowledge_fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        normalized_name TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'ACTIVE'
            CHECK(status IN ('ACTIVE', 'DISABLED', 'RETRACTED')),
        current_revision INTEGER NOT NULL DEFAULT 1 CHECK(current_revision >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(profile_id, instance_id, normalized_name),
        UNIQUE(profile_id, instance_id, knowledge_fact_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE knowledge_fact_revision_sources (
        knowledge_fact_id INTEGER NOT NULL,
        revision INTEGER NOT NULL,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        message_id INTEGER NOT NULL,
        quote TEXT NOT NULL,
        PRIMARY KEY(knowledge_fact_id, revision, message_id, quote),
        FOREIGN KEY(knowledge_fact_id, revision)
            REFERENCES knowledge_fact_revisions(knowledge_fact_id, revision) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id) ON DELETE CASCADE
    );

CREATE TABLE knowledge_fact_revisions (
        knowledge_fact_revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
        knowledge_fact_id INTEGER NOT NULL
            REFERENCES knowledge_fact_entries(knowledge_fact_id) ON DELETE CASCADE,
        revision INTEGER NOT NULL CHECK(revision >= 1),
        name TEXT NOT NULL,
        aliases_json TEXT NOT NULL DEFAULT '[]',
        definition TEXT NOT NULL,
        brief TEXT NOT NULL,
        importance REAL NOT NULL CHECK(importance >= 0 AND importance <= 1),
        category TEXT NOT NULL,
        session_specific_reason TEXT NOT NULL,
        origin TEXT NOT NULL DEFAULT 'KNOWLEDGE_FORMATION'
            CHECK(origin IN ('KNOWLEDGE_FORMATION', 'ADMIN')),
        change_reason TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL, valid_from TEXT, valid_until TEXT,
        UNIQUE(knowledge_fact_id, revision)
    );

CREATE TABLE knowledge_fact_terms (
        knowledge_fact_id INTEGER NOT NULL
            REFERENCES knowledge_fact_entries(knowledge_fact_id) ON DELETE CASCADE,
        revision INTEGER NOT NULL,
        term TEXT NOT NULL,
        normalized_term TEXT NOT NULL,
        term_kind TEXT NOT NULL CHECK(term_kind IN ('NAME', 'ALIAS', 'KEYWORD')),
        created_at TEXT NOT NULL,
        PRIMARY KEY(knowledge_fact_id, revision, term_kind, normalized_term),
        FOREIGN KEY(knowledge_fact_id, revision)
            REFERENCES knowledge_fact_revisions(knowledge_fact_id, revision) ON DELETE CASCADE
    );

CREATE TABLE knowledge_message_marks (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        message_id INTEGER NOT NULL,
        outcome TEXT NOT NULL CHECK(outcome IN (
            'PROCESSED', 'NO_KNOWLEDGE', 'TERMINAL_EXCLUDED'
        )),
        batch_id INTEGER REFERENCES knowledge_batches(batch_id) ON DELETE SET NULL,
        reason TEXT NOT NULL DEFAULT '',
        marked_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id, message_id),
        FOREIGN KEY(profile_id, instance_id, message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id) ON DELETE CASCADE
    );

CREATE TABLE knowledge_processing_state (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        baseline_message_id INTEGER NOT NULL DEFAULT 0,
        committed_through_message_id INTEGER NOT NULL DEFAULT 0,
        desired_through_message_id INTEGER NOT NULL DEFAULT 0,
        active_task_id INTEGER REFERENCES "ai_tasks"(task_id) ON DELETE SET NULL,
        processing_version INTEGER NOT NULL DEFAULT 1 CHECK(processing_version >= 1),
        last_message_at TEXT,
        last_trigger_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE main_core_work_checkpoint_events (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        work_ref TEXT NOT NULL,
        event_sequence INTEGER NOT NULL CHECK(event_sequence >= 1),
        event_kind TEXT NOT NULL CHECK(event_kind IN (
            'CREATED', 'CALLBACK_ACCEPTED', 'CALLBACK_REJECTED',
            'LEASE_CLAIMED', 'LEASE_RENEWED', 'LEASE_RELEASED',
            'CANCELLED', 'EXPIRED', 'SUPERSEDED'
        )),
        checkpoint_version INTEGER NOT NULL CHECK(checkpoint_version >= 1),
        run_generation INTEGER NOT NULL CHECK(run_generation >= 1),
        callback_sequence INTEGER NOT NULL CHECK(callback_sequence >= 0),
        checkpoint_status TEXT NOT NULL CHECK(checkpoint_status IN (
            'WAITING', 'RECOVERY_READY', 'CANCELLED', 'EXPIRED', 'SUPERSEDED'
        )),
        idempotency_key TEXT NOT NULL CHECK(length(idempotency_key) BETWEEN 1 AND 160),
        request_fingerprint TEXT NOT NULL CHECK(length(request_fingerprint) = 64),
        decision_code TEXT NOT NULL DEFAULT '' CHECK(length(decision_code) <= 64),
        created_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id, work_ref, event_sequence),
        FOREIGN KEY(profile_id, instance_id, work_ref)
            REFERENCES main_core_work_checkpoints(profile_id, instance_id, work_ref)
            ON DELETE CASCADE
    );

CREATE TABLE main_core_work_checkpoint_receipts (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        work_ref TEXT NOT NULL,
        idempotency_key TEXT NOT NULL CHECK(length(idempotency_key) BETWEEN 1 AND 160),
        operation_kind TEXT NOT NULL CHECK(length(operation_kind) BETWEEN 1 AND 64),
        request_fingerprint TEXT NOT NULL CHECK(length(request_fingerprint) = 64),
        result_json TEXT NOT NULL CHECK(length(result_json) BETWEEN 1 AND 192000),
        created_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id, work_ref, idempotency_key),
        FOREIGN KEY(profile_id, instance_id, work_ref)
            REFERENCES main_core_work_checkpoints(profile_id, instance_id, work_ref)
            ON DELETE CASCADE
    );

CREATE TABLE main_core_work_checkpoints (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        work_ref TEXT NOT NULL CHECK(length(work_ref) BETWEEN 1 AND 160),
        checkpoint_json TEXT NOT NULL CHECK(length(checkpoint_json) BETWEEN 1 AND 96000),
        recovery_envelope_json TEXT CHECK(
            recovery_envelope_json IS NULL
            OR length(recovery_envelope_json) BETWEEN 1 AND 96000
        ),
        status TEXT NOT NULL CHECK(status IN (
            'WAITING', 'RECOVERY_READY', 'CANCELLED', 'EXPIRED', 'SUPERSEDED'
        )),
        checkpoint_version INTEGER NOT NULL CHECK(checkpoint_version >= 1),
        run_generation INTEGER NOT NULL CHECK(run_generation >= 1),
        callback_sequence INTEGER NOT NULL CHECK(callback_sequence >= 0),
        lease_owner TEXT,
        lease_token INTEGER,
        lease_expires_at TEXT,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        terminal_reason TEXT NOT NULL DEFAULT '' CHECK(length(terminal_reason) <= 400),
        last_idempotency_key TEXT NOT NULL DEFAULT ''
            CHECK(length(last_idempotency_key) <= 160),
        last_callback_fingerprint TEXT NOT NULL DEFAULT '' CHECK(
            last_callback_fingerprint = '' OR length(last_callback_fingerprint) = 64
        ),
        PRIMARY KEY(profile_id, instance_id, work_ref),
        CHECK((status = 'WAITING') = (lease_owner IS NOT NULL)),
        CHECK((lease_owner IS NULL) = (lease_token IS NULL)),
        CHECK((lease_owner IS NULL) = (lease_expires_at IS NULL)),
        CHECK((status = 'RECOVERY_READY') = (recovery_envelope_json IS NOT NULL)),
        CHECK((last_idempotency_key = '') = (last_callback_fingerprint = '')),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE main_core_work_file_bindings (
        job_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        work_ref TEXT NOT NULL CHECK(length(work_ref) BETWEEN 1 AND 160),
        request_ref TEXT NOT NULL CHECK(length(request_ref) BETWEEN 1 AND 160),
        slot_id TEXT NOT NULL CHECK(length(slot_id) BETWEEN 1 AND 160),
        checkpoint_version INTEGER NOT NULL CHECK(checkpoint_version >= 1),
        run_generation INTEGER NOT NULL CHECK(run_generation >= 1),
        callback_sequence INTEGER NOT NULL CHECK(callback_sequence >= 1),
        status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN (
            'PENDING', 'SUCCEEDED', 'FAILED', 'CANCELLED'
        )),
        resource_ref TEXT NOT NULL DEFAULT '' CHECK(length(resource_ref) <= 160),
        result_kind TEXT NOT NULL DEFAULT '' CHECK(length(result_kind) <= 64),
        result_summary TEXT NOT NULL DEFAULT '' CHECK(length(result_summary) <= 1000),
        todo_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT,
        UNIQUE(profile_id, instance_id, work_ref, request_ref),
        CHECK((status = 'PENDING') = (completed_at IS NULL)),
        CHECK(status = 'PENDING' OR result_kind <> ''),
        CHECK(status <> 'SUCCEEDED' OR resource_ref <> ''),
        FOREIGN KEY(job_id) REFERENCES file_generation_jobs(job_id) ON DELETE RESTRICT,
        FOREIGN KEY(profile_id, instance_id, work_ref)
            REFERENCES main_core_work_checkpoints(profile_id, instance_id, work_ref)
            ON DELETE CASCADE,
        FOREIGN KEY(todo_id) REFERENCES important_todos(todo_id) ON DELETE RESTRICT
    );

CREATE TABLE main_core_work_recovery_wakes (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        work_ref TEXT NOT NULL CHECK(length(work_ref) BETWEEN 1 AND 160),
        checkpoint_version INTEGER NOT NULL CHECK(checkpoint_version >= 1),
        wakeup_id INTEGER NOT NULL UNIQUE,
        status TEXT NOT NULL CHECK(status IN (
            'READY', 'CLAIMED', 'EXPIRED', 'SUPERSEDED'
        )),
        claimed_run_id INTEGER,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        claimed_at TEXT,
        terminal_reason TEXT NOT NULL DEFAULT '' CHECK(length(terminal_reason) <= 400),
        PRIMARY KEY(profile_id, instance_id, work_ref, checkpoint_version),
        CHECK((status = 'CLAIMED') = (claimed_run_id IS NOT NULL)),
        CHECK((status = 'CLAIMED') = (claimed_at IS NOT NULL)),
        FOREIGN KEY(profile_id, instance_id, work_ref)
            REFERENCES main_core_work_checkpoints(profile_id, instance_id, work_ref)
            ON DELETE CASCADE,
        FOREIGN KEY(wakeup_id) REFERENCES instance_wakeups(wakeup_id) ON DELETE RESTRICT,
        FOREIGN KEY(profile_id, instance_id, claimed_run_id)
            REFERENCES instance_core_runs(profile_id, instance_id, run_id)
            ON DELETE RESTRICT
    );

CREATE TABLE media_asset_message_links (
        asset_id TEXT NOT NULL REFERENCES media_assets(asset_id) ON DELETE CASCADE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        message_id INTEGER NOT NULL,
        relation TEXT NOT NULL DEFAULT 'ATTACHMENT' CHECK(relation IN (
            'ATTACHMENT', 'REFERENCE', 'GENERATED_OUTPUT'
        )),
        ordinal INTEGER NOT NULL DEFAULT 0 CHECK(ordinal >= 0),
        created_at TEXT NOT NULL,
        PRIMARY KEY(asset_id, message_id, relation),
        FOREIGN KEY(profile_id, instance_id, message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id) ON DELETE CASCADE
    );

CREATE TABLE media_assets (
        asset_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        origin TEXT NOT NULL CHECK(origin IN (
            'USER_INPUT', 'GENERATED', 'STICKER_RESERVED'
        )),
        purpose TEXT NOT NULL CHECK(purpose IN (
            'NORMAL_IMAGE', 'GENERATED_IMAGE', 'STICKER'
        )),
        mime_type TEXT NOT NULL,
        file_extension TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        byte_size INTEGER NOT NULL CHECK(byte_size >= 0),
        width INTEGER CHECK(width IS NULL OR width > 0),
        height INTEGER CHECK(height IS NULL OR height > 0),
        frame_count INTEGER CHECK(frame_count IS NULL OR frame_count > 0),
        storage_relpath TEXT,
        file_status TEXT NOT NULL DEFAULT 'AVAILABLE' CHECK(file_status IN (
            'AVAILABLE', 'RELEASE_PENDING', 'RELEASED', 'MISSING', 'QUARANTINED'
        )),
        delivery_status TEXT NOT NULL DEFAULT 'NOT_SENT',
        inspection_status TEXT NOT NULL DEFAULT 'PENDING' CHECK(inspection_status IN (
            'PENDING', 'RUNNING', 'READY', 'FAILED', 'NOT_REQUIRED'
        )),
        current_projection_version INTEGER NOT NULL DEFAULT 0
            CHECK(current_projection_version >= 0),
        core_run_id INTEGER,
        ai_task_id INTEGER REFERENCES "ai_tasks"(task_id) ON DELETE SET NULL,
        summary_covered_by INTEGER REFERENCES dialogue_summaries(summary_id) ON DELETE SET NULL,
        expires_at TEXT,
        released_at TEXT,
        last_error TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(profile_id, instance_id, asset_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE media_cleanup_events (
        cleanup_id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_id TEXT REFERENCES media_assets(asset_id) ON DELETE SET NULL,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        action TEXT NOT NULL,
        status TEXT NOT NULL,
        reason TEXT NOT NULL DEFAULT '',
        details_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE media_projections (
        projection_id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_id TEXT NOT NULL REFERENCES media_assets(asset_id) ON DELETE CASCADE,
        version INTEGER NOT NULL CHECK(version >= 1),
        status TEXT NOT NULL CHECK(status IN ('PENDING', 'RUNNING', 'READY', 'FAILED')),
        visible_facts TEXT NOT NULL DEFAULT '',
        history_projection TEXT NOT NULL DEFAULT '',
        ocr_text TEXT NOT NULL DEFAULT '',
        backend_id TEXT NOT NULL DEFAULT '',
        model_id TEXT NOT NULL DEFAULT '',
        ai_task_id INTEGER REFERENCES "ai_tasks"(task_id) ON DELETE SET NULL,
        error TEXT,
        created_at TEXT NOT NULL,
        UNIQUE(asset_id, version)
    );

CREATE TABLE media_retention_holds (
        hold_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        asset_id TEXT NOT NULL REFERENCES media_assets(asset_id) ON DELETE CASCADE,
        holder_kind TEXT NOT NULL,
        holder_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        released_at TEXT,
        UNIQUE(asset_id, holder_kind, holder_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE memories (
        memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'ACTIVE'
            CHECK(status IN ('ACTIVE', 'DISABLED', 'RETRACTED')),
        current_revision INTEGER NOT NULL DEFAULT 1 CHECK(current_revision >= 1),
        content_fingerprint TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(profile_id, instance_id, content_fingerprint),
        UNIQUE(profile_id, instance_id, memory_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE memory_revision_sources (
        memory_id INTEGER NOT NULL,
        revision INTEGER NOT NULL,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        source_kind TEXT NOT NULL CHECK(source_kind = 'MESSAGE'),
        source_key TEXT NOT NULL,
        message_id INTEGER NOT NULL,
        quote TEXT NOT NULL DEFAULT '',
        source_snapshot TEXT NOT NULL DEFAULT '',
        occurred_at TEXT,
        PRIMARY KEY(memory_id, revision, source_kind, source_key),
        CHECK(quote <> ''),
        FOREIGN KEY(memory_id, revision)
            REFERENCES memory_revisions(memory_id, revision) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, message_id)
            REFERENCES instance_messages(profile_id, instance_id, message_id) ON DELETE CASCADE
    );

CREATE TABLE memory_revisions (
        memory_revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
        revision INTEGER NOT NULL CHECK(revision >= 1),
        brief TEXT NOT NULL,
        ultra_brief TEXT,
        importance REAL NOT NULL CHECK(importance >= 0 AND importance <= 1),
        event_time TEXT,
        origin TEXT NOT NULL DEFAULT 'KNOWLEDGE_FORMATION'
            CHECK(origin IN ('KNOWLEDGE_FORMATION', 'ADMIN')),
        change_reason TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        UNIQUE(memory_id, revision)
    );

CREATE TABLE memory_terms (
        memory_id INTEGER NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
        revision INTEGER NOT NULL,
        term TEXT NOT NULL,
        normalized_term TEXT NOT NULL,
        term_kind TEXT NOT NULL DEFAULT 'KEYWORD' CHECK(term_kind = 'KEYWORD'),
        created_at TEXT NOT NULL,
        PRIMARY KEY(memory_id, revision, normalized_term),
        FOREIGN KEY(memory_id, revision)
            REFERENCES memory_revisions(memory_id, revision) ON DELETE CASCADE
    );

CREATE TABLE message_retraction_actions (
        action_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        source_run_id INTEGER NOT NULL,
        expression_batch_id TEXT NOT NULL,
        step_ordinal INTEGER NOT NULL CHECK(step_ordinal >= 0),
        target_message_ref TEXT,
        target_output_ordinal INTEGER CHECK(
            target_output_ordinal IS NULL OR target_output_ordinal >= 1
        ),
        delay_after_previous_seconds INTEGER NOT NULL DEFAULT 0
            CHECK(delay_after_previous_seconds BETWEEN 0 AND 120),
        not_before_at TEXT,
        status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN (
            'PENDING', 'SENDING', 'RETRACTED', 'FAILED',
            'UNKNOWN_AFTER_CRASH', 'CANCELLED'
        )),
        idempotency_key TEXT NOT NULL,
        attempted_at TEXT,
        completed_at TEXT,
        error_code TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(profile_id, instance_id, idempotency_key),
        UNIQUE(profile_id, instance_id, expression_batch_id, step_ordinal),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, source_run_id)
            REFERENCES instance_core_runs(profile_id, instance_id, run_id),
        FOREIGN KEY(profile_id, instance_id, expression_batch_id)
            REFERENCES instance_expression_batches(profile_id, instance_id, batch_id)
            ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, target_message_ref)
            REFERENCES instance_message_fragments(profile_id, instance_id, message_ref),
        CHECK((target_message_ref IS NOT NULL) != (target_output_ordinal IS NOT NULL))
    );

CREATE TABLE message_retraction_fragment_attempts (
        action_id INTEGER NOT NULL,
        message_ref TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN (
            'PENDING', 'SENDING', 'RETRACTED', 'FAILED',
            'UNKNOWN_AFTER_CRASH', 'CANCELLED'
        )),
        attempted_at TEXT,
        completed_at TEXT,
        error_code TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(action_id, message_ref),
        FOREIGN KEY(action_id) REFERENCES message_retraction_actions(action_id)
            ON DELETE CASCADE,
        FOREIGN KEY(message_ref) REFERENCES instance_message_fragments(message_ref)
            ON DELETE CASCADE
    );

CREATE TABLE platform_connection_policies (
        profile_id TEXT NOT NULL,
        scope TEXT NOT NULL CHECK(scope IN ('private', 'group')),
        platform_instance_id TEXT NOT NULL,
        template_id TEXT NOT NULL DEFAULT 'AUTO',
        proactive_enabled INTEGER CHECK(proactive_enabled IN (0, 1)),
        check_min_minutes INTEGER CHECK(check_min_minutes IS NULL OR check_min_minutes >= 1),
        check_max_minutes INTEGER CHECK(check_max_minutes IS NULL OR check_max_minutes >= 1),
        quiet_enabled INTEGER CHECK(quiet_enabled IN (0, 1)),
        quiet_start TEXT,
        quiet_end TEXT,
        min_success_gap_minutes INTEGER CHECK(min_success_gap_minutes IS NULL OR min_success_gap_minutes >= 0),
        daily_limit_mode TEXT NOT NULL DEFAULT 'INHERIT'
            CHECK(daily_limit_mode IN ('INHERIT', 'LIMITED', 'UNLIMITED')),
        daily_success_limit INTEGER,
        unanswered_limit_mode TEXT NOT NULL DEFAULT 'INHERIT'
            CHECK(unanswered_limit_mode IN ('INHERIT', 'LIMITED', 'UNLIMITED')),
        max_consecutive_unanswered INTEGER,
        failure_mode TEXT CHECK(failure_mode IS NULL OR failure_mode IN ('SKIP', 'RETRY_BACKOFF')),
        retry_delay_minutes INTEGER CHECK(retry_delay_minutes IS NULL OR retry_delay_minutes >= 1),
        retry_max_attempts INTEGER CHECK(retry_max_attempts IS NULL OR retry_max_attempts >= 0),
        group_send_qpm_limit INTEGER CHECK(group_send_qpm_limit IS NULL OR group_send_qpm_limit >= 1),
        account_send_qpm_limit INTEGER CHECK(account_send_qpm_limit IS NULL OR account_send_qpm_limit >= 1),
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL, send_qpm_limit INTEGER
        CHECK(send_qpm_limit IS NULL OR send_qpm_limit >= 1),
        CHECK((daily_limit_mode = 'LIMITED' AND daily_success_limit >= 1)
            OR (daily_limit_mode IN ('INHERIT', 'UNLIMITED') AND daily_success_limit IS NULL)),
        CHECK((unanswered_limit_mode = 'LIMITED' AND max_consecutive_unanswered >= 1)
            OR (unanswered_limit_mode IN ('INHERIT', 'UNLIMITED') AND max_consecutive_unanswered IS NULL)),
        PRIMARY KEY(profile_id, scope, platform_instance_id),
        FOREIGN KEY(profile_id, scope) REFERENCES scope_configs(profile_id, scope)
            ON DELETE CASCADE
    );

CREATE TABLE platform_message_media_refs (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        platform_message_id TEXT NOT NULL,
        asset_id TEXT NOT NULL REFERENCES media_assets(asset_id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL DEFAULT 0 CHECK(ordinal >= 0),
        created_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id, platform_message_id, asset_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE platform_send_permits (
        permit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        platform_instance_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        account_key TEXT NOT NULL DEFAULT '',
        origin_kind TEXT NOT NULL,
        origin_id TEXT NOT NULL,
        fragment_index INTEGER NOT NULL CHECK(fragment_index >= 0),
        status TEXT NOT NULL DEFAULT 'RESERVED' CHECK(status IN (
            'RESERVED', 'DISPATCHING', 'ATTEMPTED_UNKNOWN',
            'RELEASED', 'FAILED_BEFORE_DISPATCH'
        )),
        reservation_key TEXT NOT NULL UNIQUE,
        reserved_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        lease_until TEXT NOT NULL,
        dispatched_at TEXT,
        detail TEXT NOT NULL DEFAULT '',
        updated_at TEXT NOT NULL
    );

CREATE TABLE player_profile_command_receipts (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        subject_key TEXT NOT NULL,
        idempotency_key TEXT NOT NULL CHECK(length(idempotency_key) BETWEEN 1 AND 200),
        command_fingerprint TEXT NOT NULL CHECK(length(command_fingerprint) = 64),
        result_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id, subject_key, idempotency_key),
        FOREIGN KEY(profile_id, instance_id, subject_key)
            REFERENCES player_profiles(profile_id, instance_id, subject_key) ON DELETE CASCADE
    );

CREATE TABLE player_profile_entries (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        subject_key TEXT NOT NULL,
        entry_id TEXT NOT NULL,
        current_entry_version INTEGER NOT NULL CHECK(current_entry_version >= 1),
        PRIMARY KEY(profile_id, instance_id, subject_key, entry_id),
        FOREIGN KEY(profile_id, instance_id, subject_key)
            REFERENCES player_profiles(profile_id, instance_id, subject_key) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, subject_key, entry_id, current_entry_version)
            REFERENCES player_profile_entry_revisions(
                profile_id, instance_id, subject_key, entry_id, entry_version
            ) ON DELETE CASCADE
    );

CREATE TABLE player_profile_entry_revisions (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        subject_key TEXT NOT NULL,
        entry_id TEXT NOT NULL CHECK(length(entry_id) BETWEEN 1 AND 200),
        entry_version INTEGER NOT NULL CHECK(entry_version >= 1),
        layer TEXT NOT NULL CHECK(layer IN ('PLAYER_FACT', 'AI_OBSERVATION')),
        category TEXT NOT NULL CHECK(category IN (
            'SELF_DESCRIPTION', 'LIKE', 'DISLIKE', 'INTEREST', 'HABIT',
            'COMMUNICATION_PREFERENCE', 'BOUNDARY', 'AVOID_TOPIC',
            'RELATIONSHIP_NAME', 'ALIAS', 'INSTANCE_ROLE', 'LITERARY_IMPRESSION', 'OTHER'
        )),
        text TEXT NOT NULL CHECK(length(text) BETWEEN 1 AND 1000),
        source_type TEXT NOT NULL CHECK(source_type IN (
            'PLAYER_STATEMENT', 'STRONG_MESSAGE_EVIDENCE',
            'PLAYER_CORRECTION', 'AI_OBSERVATION'
        )),
        evidence_json TEXT NOT NULL,
        confidence REAL,
        sensitivity TEXT NOT NULL CHECK(sensitivity IN ('NORMAL', 'PRIVATE', 'SENSITIVE')),
        status TEXT NOT NULL CHECK(status IN ('ACTIVE', 'WITHDRAWN')),
        confirmed_at TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        withdrawal_evidence_json TEXT NOT NULL DEFAULT '[]',
        withdrawn_at TEXT,
        CHECK(
            (layer = 'PLAYER_FACT' AND source_type <> 'AI_OBSERVATION' AND confidence IS NULL)
            OR (layer = 'AI_OBSERVATION' AND source_type = 'AI_OBSERVATION'
                AND confidence BETWEEN 0.0 AND 1.0)
        ),
        CHECK(
            (status = 'ACTIVE' AND withdrawn_at IS NULL)
            OR (status = 'WITHDRAWN' AND withdrawn_at IS NOT NULL)
        ),
        PRIMARY KEY(profile_id, instance_id, subject_key, entry_id, entry_version),
        FOREIGN KEY(profile_id, instance_id, subject_key)
            REFERENCES player_profiles(profile_id, instance_id, subject_key) ON DELETE CASCADE
    );

CREATE TABLE player_profiles (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        subject_key TEXT NOT NULL CHECK(length(subject_key) BETWEEN 1 AND 200),
        current_version INTEGER NOT NULL CHECK(current_version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id, subject_key),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE profile_runtime_settings (
        profile_id TEXT PRIMARY KEY REFERENCES role_profiles(profile_id)
            ON DELETE CASCADE,
        timezone TEXT NOT NULL DEFAULT '',
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

CREATE TABLE recall_documents (
        document_key TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        source_type TEXT NOT NULL CHECK(source_type IN (
            'MEMORY', 'WORLD_INFO', 'ROLE_EVENT', 'ROLE_CURRENT',
            'DIALOGUE_SUMMARY', 'MESSAGE'
        )),
        source_key TEXT NOT NULL,
        source_revision INTEGER NOT NULL DEFAULT 1 CHECK(source_revision >= 0),
        authority_status TEXT NOT NULL CHECK(authority_status IN ('CURRENT', 'HISTORICAL')),
        title TEXT NOT NULL DEFAULT '',
        content TEXT NOT NULL,
        search_text TEXT NOT NULL,
        entity_names_json TEXT NOT NULL DEFAULT '[]',
        valid_from TEXT,
        valid_until TEXT,
        recorded_from TEXT,
        recorded_until TEXT,
        occurred_at TEXT,
        evidence_json TEXT NOT NULL DEFAULT '[]',
        source_fingerprint TEXT NOT NULL CHECK(length(source_fingerprint) = 64),
        dense_eligible INTEGER NOT NULL DEFAULT 1 CHECK(dense_eligible IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(profile_id, instance_id, source_type, source_key, source_revision),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE VIRTUAL TABLE recall_documents_fts USING fts5(
        tokens,
        document_key UNINDEXED,
        profile_id UNINDEXED,
        instance_id UNINDEXED,
        tokenize='unicode61 remove_diacritics 2'
    );

CREATE TABLE recall_edges (
        edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        source_document_key TEXT NOT NULL
            REFERENCES recall_documents(document_key) ON DELETE CASCADE,
        target_document_key TEXT NOT NULL
            REFERENCES recall_documents(document_key) ON DELETE CASCADE,
        edge_type TEXT NOT NULL CHECK(edge_type IN (
            'EVIDENCE_FOR', 'REVISED_BY', 'SUPERSEDED_BY', 'CONFLICTS_WITH',
            'BEFORE', 'AFTER', 'DERIVED_CURRENT_STATE'
        )),
        weight REAL NOT NULL DEFAULT 1.0 CHECK(weight > 0 AND weight <= 1),
        status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE', 'INACTIVE')),
        valid_from TEXT,
        valid_until TEXT,
        evidence_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        UNIQUE(source_document_key, target_document_key, edge_type),
        CHECK(source_document_key <> target_document_key),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE recall_embeddings (
        document_key TEXT NOT NULL REFERENCES recall_documents(document_key) ON DELETE CASCADE,
        generation_id INTEGER NOT NULL
            REFERENCES recall_index_generations(generation_id) ON DELETE CASCADE,
        vector_dimension INTEGER NOT NULL CHECK(vector_dimension >= 1),
        vector_blob BLOB NOT NULL,
        content_hash TEXT NOT NULL CHECK(length(content_hash) = 64),
        provider_fingerprint TEXT NOT NULL CHECK(length(provider_fingerprint) = 64),
        created_at TEXT NOT NULL,
        PRIMARY KEY(document_key, generation_id)
    );

CREATE TABLE recall_graph_edges (
        edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        source_node_key TEXT NOT NULL
            REFERENCES recall_graph_nodes(node_key) ON DELETE CASCADE,
        target_node_key TEXT NOT NULL
            REFERENCES recall_graph_nodes(node_key) ON DELETE CASCADE,
        edge_type TEXT NOT NULL CHECK(edge_type IN (
            'EVIDENCE_FOR', 'PARTICIPATED_IN', 'MENTIONS_ENTITY',
            'BEFORE', 'AFTER', 'REVISED_BY', 'SUPERSEDED_BY',
            'CONFLICTS_WITH', 'DERIVED_CURRENT_STATE',
            'BELONGS_TO_SCENE', 'BELONGS_TO_TOPIC'
        )),
        weight REAL NOT NULL DEFAULT 1.0 CHECK(weight > 0 AND weight <= 1),
        status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE', 'INACTIVE')),
        evidence_document_key TEXT
            REFERENCES recall_documents(document_key) ON DELETE CASCADE,
        valid_from TEXT,
        valid_until TEXT,
        evidence_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        UNIQUE(source_node_key, target_node_key, edge_type),
        CHECK(source_node_key <> target_node_key),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE recall_graph_nodes (
        node_key TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        node_type TEXT NOT NULL CHECK(node_type IN (
            'PERSON', 'ENTITY', 'EVENT', 'FACT', 'SCENE', 'EVIDENCE'
        )),
        stable_ref TEXT NOT NULL,
        label TEXT NOT NULL DEFAULT '',
        document_key TEXT REFERENCES recall_documents(document_key) ON DELETE CASCADE,
        scene_key TEXT REFERENCES recall_scenes(scene_key) ON DELETE CASCADE,
        valid_from TEXT,
        valid_until TEXT,
        evidence_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        UNIQUE(profile_id, instance_id, node_type, stable_ref),
        CHECK(document_key IS NOT NULL OR scene_key IS NOT NULL OR node_type IN ('PERSON', 'ENTITY')),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE recall_index_generations (
        generation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        embedding_provider_id TEXT NOT NULL,
        provider_fingerprint TEXT NOT NULL CHECK(length(provider_fingerprint) = 64),
        vector_dimension INTEGER NOT NULL CHECK(vector_dimension >= 1),
        status TEXT NOT NULL DEFAULT 'BUILDING'
            CHECK(status IN ('BUILDING', 'READY', 'FAILED', 'RETIRED')),
        active INTEGER NOT NULL DEFAULT 0 CHECK(active IN (0, 1)),
        document_count INTEGER NOT NULL DEFAULT 0 CHECK(document_count >= 0),
        embedded_count INTEGER NOT NULL DEFAULT 0 CHECK(embedded_count >= 0),
        failure_reason TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        activated_at TEXT,
        UNIQUE(profile_id, instance_id, provider_fingerprint),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE recall_index_outbox (
        task_id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_key TEXT NOT NULL UNIQUE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        source_type TEXT NOT NULL CHECK(source_type IN (
            'MEMORY', 'WORLD_INFO', 'ROLE_EVENT', 'ROLE_CURRENT',
            'DIALOGUE_SUMMARY', 'MESSAGE', 'SCOPE'
        )),
        source_key TEXT NOT NULL,
        operation TEXT NOT NULL CHECK(operation IN ('UPSERT', 'DELETE', 'REBUILD')),
        source_version INTEGER NOT NULL DEFAULT 0 CHECK(source_version >= 0),
        status TEXT NOT NULL DEFAULT 'PENDING'
            CHECK(status IN ('PENDING', 'LEASED', 'COMPLETED', 'FAILED')),
        lease_owner TEXT,
        lease_token INTEGER NOT NULL DEFAULT 0 CHECK(lease_token >= 0),
        lease_until TEXT,
        attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
        last_error TEXT NOT NULL DEFAULT '',
        not_before TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE "recall_probe_reports" (
        report_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        current_message_id INTEGER,
        query TEXT NOT NULL DEFAULT '',
        report_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE recall_role_settings (
        profile_id TEXT PRIMARY KEY REFERENCES role_profiles(profile_id) ON DELETE CASCADE,
        embedding_provider_id TEXT,
        rerank_provider_id TEXT,
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        updated_at TEXT NOT NULL
    );

CREATE TABLE recall_scene_members (
        scene_key TEXT NOT NULL REFERENCES recall_scenes(scene_key) ON DELETE CASCADE,
        document_key TEXT NOT NULL REFERENCES recall_documents(document_key) ON DELETE CASCADE,
        membership_weight REAL NOT NULL DEFAULT 1.0 CHECK(membership_weight > 0 AND membership_weight <= 1),
        evidence_json TEXT NOT NULL DEFAULT '[]',
        PRIMARY KEY(scene_key, document_key)
    );

CREATE TABLE recall_scenes (
        scene_key TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        parent_scene_key TEXT REFERENCES recall_scenes(scene_key) ON DELETE SET NULL,
        scene_level TEXT NOT NULL CHECK(scene_level IN ('EVENT', 'TOPIC')),
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        search_text TEXT NOT NULL,
        occurred_from TEXT,
        occurred_until TEXT,
        evidence_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE role_profiles (
        profile_id TEXT PRIMARY KEY,
        name TEXT NOT NULL DEFAULT '',
        enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0, 1)),
        quick_setup_decided INTEGER NOT NULL DEFAULT 0
            CHECK(quick_setup_decided IN (0, 1)),
        thinking_complexity TEXT NOT NULL DEFAULT '标准'
            CHECK(thinking_complexity IN ('极简', '轻量', '均衡', '标准', '深入', '极致')),
        background_life_enabled INTEGER NOT NULL DEFAULT 0
            CHECK(background_life_enabled IN (0, 1)),
        background_life_version INTEGER NOT NULL DEFAULT 1
            CHECK(background_life_version >= 1),
        proactive_enabled INTEGER NOT NULL DEFAULT 1 CHECK(proactive_enabled IN (0, 1)),
        extra_background TEXT NOT NULL DEFAULT '',
        min_wakeup_minutes INTEGER NOT NULL DEFAULT 15 CHECK(min_wakeup_minutes >= 1),
        max_wakeup_minutes INTEGER NOT NULL DEFAULT 55 CHECK(max_wakeup_minutes >= min_wakeup_minutes),
        orphaned INTEGER NOT NULL DEFAULT 0 CHECK(orphaned IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    , low_frequency_min_wakeup_minutes
        INTEGER NOT NULL DEFAULT 180 CHECK(low_frequency_min_wakeup_minutes >= 1), low_frequency_max_wakeup_minutes
        INTEGER NOT NULL DEFAULT 480 CHECK(low_frequency_max_wakeup_minutes >= 1), image_generation_enabled INTEGER NOT NULL DEFAULT 0
        CHECK(image_generation_enabled IN (0, 1)), web_search_enabled INTEGER NOT NULL DEFAULT 0
        CHECK(web_search_enabled IN (0, 1)), web_search_intensity TEXT NOT NULL DEFAULT 'STANDARD'
        CHECK(web_search_intensity IN ('ECONOMY', 'STANDARD', 'DEEP')), file_artifacts_enabled
        INTEGER NOT NULL DEFAULT 0 CHECK(file_artifacts_enabled IN (0, 1)), turn_buffer_enabled
        INTEGER NOT NULL DEFAULT 1 CHECK(turn_buffer_enabled IN (0, 1)));

CREATE TABLE runtime_file_cleanup_queue (
        cleanup_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL DEFAULT '',
        storage_kind TEXT NOT NULL CHECK(storage_kind IN (
            'MEDIA', 'FILE_ARTIFACT', 'VOICE_ARTIFACT'
        )),
        storage_relpath TEXT NOT NULL CHECK(storage_relpath <> ''),
        owner_id TEXT NOT NULL DEFAULT '',
        expected_sha256 TEXT NOT NULL CHECK(length(expected_sha256) = 64),
        expected_byte_size INTEGER NOT NULL CHECK(expected_byte_size >= 0),
        reason TEXT NOT NULL CHECK(reason <> ''),
        not_before_at TEXT NOT NULL,
        attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
        last_error TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(storage_kind, storage_relpath)
    );

CREATE TABLE scope_configs (
        profile_id TEXT NOT NULL REFERENCES role_profiles(profile_id) ON DELETE CASCADE,
        scope TEXT NOT NULL CHECK(scope IN ('private', 'group')),
        proactive_enabled INTEGER NOT NULL DEFAULT 1 CHECK(proactive_enabled IN (0, 1)),
        extra_background TEXT NOT NULL DEFAULT '',
        min_wakeup_minutes INTEGER NOT NULL DEFAULT 15 CHECK(min_wakeup_minutes >= 1),
        max_wakeup_minutes INTEGER NOT NULL DEFAULT 55
            CHECK(max_wakeup_minutes >= min_wakeup_minutes),
        low_frequency_min_wakeup_minutes INTEGER NOT NULL DEFAULT 180
            CHECK(low_frequency_min_wakeup_minutes >= 1),
        low_frequency_max_wakeup_minutes INTEGER NOT NULL DEFAULT 480
            CHECK(low_frequency_max_wakeup_minutes >= low_frequency_min_wakeup_minutes),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL, max_context_tokens
        INTEGER NOT NULL DEFAULT 128000 CHECK(max_context_tokens >= 128000), target_context_tokens
        INTEGER NOT NULL DEFAULT 64000
        CHECK(target_context_tokens >= 20000
            AND target_context_tokens <= max_context_tokens), world_texture_prompt
        TEXT NOT NULL DEFAULT '', version INTEGER NOT NULL DEFAULT 1
        CHECK(version >= 1), media_original_retention_days INTEGER
        NOT NULL DEFAULT 30
        CHECK(media_original_retention_days BETWEEN 0 AND 3650),
        PRIMARY KEY(profile_id, scope)
    );

CREATE TABLE scope_delivery_policies (
        profile_id TEXT NOT NULL,
        scope TEXT NOT NULL CHECK(scope IN ('private', 'group')),
        group_send_qpm_limit INTEGER NOT NULL DEFAULT 20 CHECK(group_send_qpm_limit >= 1),
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL, send_qpm_limit INTEGER
        NOT NULL DEFAULT 20 CHECK(send_qpm_limit >= 1),
        PRIMARY KEY(profile_id, scope),
        FOREIGN KEY(profile_id, scope) REFERENCES scope_configs(profile_id, scope)
            ON DELETE CASCADE
    );

CREATE TABLE scope_state_gate_policies (
        profile_id TEXT NOT NULL,
        scope TEXT NOT NULL CHECK(scope IN ('private', 'group')),
        enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0, 1)),
        silent_enabled INTEGER NOT NULL DEFAULT 0 CHECK(silent_enabled IN (0, 1)),
        max_gate_hours INTEGER NOT NULL DEFAULT 24
            CHECK(max_gate_hours >= 1 AND max_gate_hours <= 24),
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, scope),
        FOREIGN KEY(profile_id, scope) REFERENCES scope_configs(profile_id, scope)
            ON DELETE CASCADE
    );

CREATE TABLE soulcore_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL REFERENCES role_profiles(profile_id) ON DELETE CASCADE,
        instance_id TEXT,
        level TEXT NOT NULL,
        category TEXT NOT NULL,
        message TEXT NOT NULL,
        details_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );

CREATE TABLE sticker_assets (
        sticker_asset_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL REFERENCES role_profiles(profile_id) ON DELETE CASCADE,
        canonical_sha256 TEXT NOT NULL,
        storage_relpath TEXT NOT NULL UNIQUE,
        mime_type TEXT NOT NULL,
        file_extension TEXT NOT NULL,
        byte_size INTEGER NOT NULL CHECK(byte_size > 0),
        width INTEGER NOT NULL CHECK(width > 0),
        height INTEGER NOT NULL CHECK(height > 0),
        is_animated INTEGER NOT NULL DEFAULT 0 CHECK(is_animated IN (0, 1)),
        frame_count INTEGER NOT NULL DEFAULT 1 CHECK(frame_count >= 1),
        duration_ms INTEGER NOT NULL DEFAULT 0 CHECK(duration_ms >= 0),
        file_status TEXT NOT NULL DEFAULT 'AVAILABLE'
            CHECK(file_status IN ('AVAILABLE', 'RELEASE_PENDING', 'RELEASED', 'MISSING')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(profile_id, canonical_sha256)
    );

CREATE TABLE sticker_candidates (
        candidate_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        target_library_id TEXT NOT NULL REFERENCES sticker_libraries(library_id)
            ON DELETE CASCADE,
        source_kind TEXT NOT NULL CHECK(source_kind IN ('PLAYER', 'WEB', 'GENERATED', 'UPLOAD')),
        source_asset_id TEXT NOT NULL REFERENCES media_assets(asset_id) ON DELETE RESTRICT,
        source_ref TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN (
            'PENDING', 'CHECKING', 'WAITING_CHECK', 'READY',
            'ACCEPTED', 'REJECTED', 'QUARANTINED'
        )),
        import_count INTEGER NOT NULL DEFAULT 1 CHECK(import_count >= 1),
        persona_fingerprint TEXT NOT NULL DEFAULT '',
        accepted_item_id TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        last_error TEXT NOT NULL DEFAULT '',
        failure_stage TEXT NOT NULL DEFAULT '',
        retry_count INTEGER NOT NULL DEFAULT 0 CHECK(retry_count >= 0),
        next_retry_at TEXT,
        recoverable INTEGER NOT NULL DEFAULT 0 CHECK(recoverable IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(profile_id, instance_id, source_asset_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE sticker_check_revisions (
        check_id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id TEXT NOT NULL REFERENCES sticker_candidates(candidate_id) ON DELETE CASCADE,
        revision INTEGER NOT NULL CHECK(revision >= 1),
        verdict TEXT NOT NULL CHECK(verdict IN ('ACCEPT', 'REJECT', 'QUARANTINE')),
        compact_name TEXT NOT NULL DEFAULT '',
        compact_description TEXT NOT NULL DEFAULT '',
        visible_text TEXT NOT NULL DEFAULT '',
        usage_type TEXT NOT NULL DEFAULT 'REACTION'
            CHECK(usage_type IN ('AMBIENT', 'REACTION', 'SPECIFIC')),
        semantic_key TEXT NOT NULL DEFAULT '',
        emotion TEXT NOT NULL DEFAULT '',
        speech_act TEXT NOT NULL DEFAULT '',
        intensity INTEGER NOT NULL DEFAULT 0 CHECK(intensity BETWEEN 0 AND 10),
        persona_score REAL NOT NULL DEFAULT 0,
        reason TEXT NOT NULL DEFAULT '',
        backend_id TEXT NOT NULL DEFAULT '',
        model_id TEXT NOT NULL DEFAULT '',
        details_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE(candidate_id, revision)
    );

CREATE TABLE sticker_clusters (
        cluster_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        library_id TEXT NOT NULL REFERENCES sticker_libraries(library_id) ON DELETE CASCADE,
        semantic_key TEXT NOT NULL,
        active_count INTEGER NOT NULL DEFAULT 0 CHECK(active_count >= 0),
        auto_count INTEGER NOT NULL DEFAULT 0 CHECK(auto_count >= 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(library_id, cluster_id),
        CHECK(length(instance_id) > 0)
    );

CREATE TABLE sticker_configs (
        profile_id TEXT NOT NULL,
        scope TEXT NOT NULL CHECK(scope IN ('private', 'group')),
        enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0, 1)),
        player_collection_enabled INTEGER NOT NULL DEFAULT 0 CHECK(player_collection_enabled IN (0, 1)),
        web_collection_enabled INTEGER NOT NULL DEFAULT 0 CHECK(web_collection_enabled IN (0, 1)),
        generation_enabled INTEGER NOT NULL DEFAULT 0 CHECK(generation_enabled IN (0, 1)),
        trigger_mode TEXT NOT NULL DEFAULT 'TURNS_ONLY' CHECK(trigger_mode IN (
            'TURNS_ONLY', 'TIME_ONLY', 'ANY', 'ALL'
        )),
        turn_threshold INTEGER NOT NULL DEFAULT 20 CHECK(turn_threshold >= 1),
        elapsed_hours REAL NOT NULL DEFAULT 24 CHECK(elapsed_hours > 0),
        library_limit INTEGER NOT NULL DEFAULT 1000 CHECK(library_limit BETWEEN 50 AND 10000),
        web_daily_limit INTEGER NOT NULL DEFAULT 4 CHECK(web_daily_limit >= 0),
        generated_daily_limit INTEGER NOT NULL DEFAULT 1 CHECK(generated_daily_limit >= 0),
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        requirements TEXT NOT NULL DEFAULT '__DEFAULT_STICKER_REQUIREMENTS__',
        PRIMARY KEY(profile_id, scope),
        FOREIGN KEY(profile_id, scope)
            REFERENCES scope_configs(profile_id, scope) ON DELETE CASCADE
    );

CREATE TABLE sticker_fingerprints (
        item_id TEXT PRIMARY KEY REFERENCES sticker_items(item_id) ON DELETE CASCADE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        library_id TEXT NOT NULL REFERENCES sticker_libraries(library_id) ON DELETE CASCADE,
        phash TEXT NOT NULL DEFAULT '',
        dhash TEXT NOT NULL DEFAULT '',
        frame_hashes_json TEXT NOT NULL DEFAULT '[]',
        representative_frame_hashes_json TEXT NOT NULL DEFAULT '[]',
        visual_group TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK(length(instance_id) > 0)
    );

CREATE TABLE sticker_import_events (
        import_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        candidate_id TEXT REFERENCES sticker_candidates(candidate_id) ON DELETE SET NULL,
        item_id TEXT REFERENCES sticker_items(item_id) ON DELETE SET NULL,
        source_kind TEXT NOT NULL CHECK(source_kind IN ('PLAYER', 'WEB', 'GENERATED', 'UPLOAD')),
        source_ref TEXT NOT NULL DEFAULT '',
        exact_duplicate INTEGER NOT NULL DEFAULT 0 CHECK(exact_duplicate IN (0, 1)),
        created_at TEXT NOT NULL,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE sticker_instance_item_states (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        disabled_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id, item_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        FOREIGN KEY(item_id)
            REFERENCES sticker_items(item_id) ON DELETE CASCADE
    );

CREATE TABLE sticker_intake_entries (
        entry_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sticker_intake_sessions(session_id)
            ON DELETE CASCADE,
        client_entry_id TEXT NOT NULL CHECK(length(client_entry_id) BETWEEN 1 AND 128),
        candidate_id TEXT REFERENCES sticker_candidates(candidate_id) ON DELETE SET NULL,
        content_sha256 TEXT NOT NULL DEFAULT ''
            CHECK(content_sha256 = '' OR length(content_sha256) = 64),
        display_name TEXT NOT NULL DEFAULT '' CHECK(length(display_name) <= 160),
        source_ref TEXT NOT NULL DEFAULT '' CHECK(length(source_ref) <= 500),
        status TEXT NOT NULL CHECK(status IN (
            'PENDING', 'UPLOADED', 'ANALYZING', 'READY', 'REJECTED',
            'DUPLICATE', 'ERROR', 'CANCELLED', 'IMPORTED'
        )),
        selected INTEGER NOT NULL DEFAULT 1 CHECK(selected IN (0, 1)),
        reason_code TEXT NOT NULL DEFAULT '' CHECK(length(reason_code) <= 100),
        error_message TEXT NOT NULL DEFAULT '' CHECK(length(error_message) <= 500),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(session_id, client_entry_id)
    );

CREATE TABLE sticker_intake_sessions (
        session_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        scope TEXT NOT NULL CHECK(scope IN ('private', 'group')),
        instance_id TEXT NOT NULL,
        intake_kind TEXT NOT NULL CHECK(intake_kind IN ('UPLOAD', 'SEARCH')),
        status TEXT NOT NULL CHECK(status IN (
            'UPLOADING', 'RUNNING', 'REVIEW', 'FINALIZING',
            'COMPLETED', 'CANCELLED', 'FAILED'
        )),
        target_count INTEGER NOT NULL CHECK(target_count BETWEEN 1 AND 50),
        raw_limit INTEGER NOT NULL CHECK(raw_limit BETWEEN 1 AND 150),
        expected_count INTEGER NOT NULL DEFAULT 0 CHECK(expected_count BETWEEN 0 AND 50),
        user_prompt TEXT NOT NULL DEFAULT '' CHECK(length(user_prompt) <= 500),
        task_id INTEGER REFERENCES ai_tasks(task_id) ON DELETE SET NULL,
        stop_requested INTEGER NOT NULL DEFAULT 0 CHECK(stop_requested IN (0, 1)),
        finalize_action TEXT NOT NULL DEFAULT ''
            CHECK(finalize_action IN ('', 'FINISH', 'CANCEL')),
        last_error TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        completed_at TEXT,
        FOREIGN KEY(profile_id, scope)
            REFERENCES scope_configs(profile_id, scope) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE,
        CHECK((status IN ('COMPLETED', 'CANCELLED')) = (completed_at IS NOT NULL))
    );

CREATE TABLE sticker_items (
        item_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        library_id TEXT NOT NULL REFERENCES sticker_libraries(library_id) ON DELETE CASCADE,
        asset_id TEXT NOT NULL REFERENCES sticker_assets(sticker_asset_id) ON DELETE RESTRICT,
        canonical_sha256 TEXT NOT NULL,
        source_kind TEXT NOT NULL CHECK(source_kind IN ('PLAYER', 'WEB', 'GENERATED', 'UPLOAD')),
        compact_name TEXT NOT NULL DEFAULT '',
        compact_description TEXT NOT NULL,
        visible_text TEXT NOT NULL DEFAULT '',
        ocr_text TEXT NOT NULL DEFAULT '',
        usage_type TEXT NOT NULL DEFAULT 'REACTION'
            CHECK(usage_type IN ('AMBIENT', 'REACTION', 'SPECIFIC')),
        vibe_tags_json TEXT NOT NULL DEFAULT '[]',
        search_keywords_json TEXT NOT NULL DEFAULT '[]',
        search_index TEXT NOT NULL DEFAULT '',
        semantic_key TEXT NOT NULL,
        cluster_id TEXT NOT NULL REFERENCES sticker_clusters(cluster_id) ON DELETE RESTRICT,
        emotion TEXT NOT NULL DEFAULT '',
        speech_act TEXT NOT NULL DEFAULT '',
        intensity INTEGER NOT NULL DEFAULT 0 CHECK(intensity BETWEEN 0 AND 10),
        persona_score REAL NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN (
            'ACTIVE', 'NEEDS_REVIEW', 'ARCHIVED', 'DELETED'
        )),
        import_count INTEGER NOT NULL DEFAULT 1 CHECK(import_count >= 1),
        reinforcement_score REAL NOT NULL DEFAULT 0,
        usage_count INTEGER NOT NULL DEFAULT 0 CHECK(usage_count >= 0),
        last_used_at TEXT,
        mime_type TEXT NOT NULL DEFAULT 'image/png',
        is_animated INTEGER NOT NULL DEFAULT 0 CHECK(is_animated IN (0, 1)),
        frame_count INTEGER NOT NULL DEFAULT 1 CHECK(frame_count >= 1),
        duration_ms INTEGER NOT NULL DEFAULT 0 CHECK(duration_ms >= 0),
        representative_frame_hashes_json TEXT NOT NULL DEFAULT '[]',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(library_id, asset_id),
        CHECK(length(instance_id) > 0)
    );

CREATE TABLE sticker_libraries (
        library_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        scope TEXT NOT NULL CHECK(scope IN ('private', 'group')),
        library_kind TEXT NOT NULL CHECK(library_kind IN ('CORE', 'PRIVATE')),
        instance_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK((library_kind = 'CORE' AND instance_id IS NULL)
           OR (library_kind = 'PRIVATE' AND instance_id IS NOT NULL)),
        FOREIGN KEY(profile_id, scope)
            REFERENCES scope_configs(profile_id, scope) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE sticker_library_states (
        library_id TEXT PRIMARY KEY REFERENCES sticker_libraries(library_id) ON DELETE CASCADE,
        normal_last_success_at TEXT,
        last_error TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

CREATE TABLE sticker_reinforcements (
        reinforcement_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        item_id TEXT NOT NULL REFERENCES sticker_items(item_id) ON DELETE CASCADE,
        run_id TEXT NOT NULL DEFAULT '',
        strength REAL NOT NULL,
        reason TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE sticker_run_candidates (
        sticker_ref TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        item_id TEXT NOT NULL REFERENCES sticker_items(item_id) ON DELETE CASCADE,
        compact_description TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(profile_id, instance_id, run_id, item_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE sticker_trigger_states (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        processed_through_message_id INTEGER NOT NULL DEFAULT 0 CHECK(processed_through_message_id >= 0),
        frozen_through_message_id INTEGER CHECK(frozen_through_message_id IS NULL OR frozen_through_message_id >= 0),
        enabled_at TEXT NOT NULL,
        last_success_at TEXT,
        cooldown_until TEXT,
        active_task_id INTEGER REFERENCES ai_tasks(task_id) ON DELETE SET NULL,
        last_error TEXT NOT NULL DEFAULT '',
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE sticker_usages (
        usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        item_id TEXT NOT NULL REFERENCES sticker_items(item_id) ON DELETE RESTRICT,
        run_id TEXT NOT NULL,
        sticker_ref TEXT NOT NULL,
        compact_projection TEXT NOT NULL,
        delivery_status TEXT NOT NULL,
        outbox_id INTEGER,
        expression_ordinal INTEGER,
        message_id INTEGER,
        created_at TEXT NOT NULL,
        CHECK((outbox_id IS NULL) = (expression_ordinal IS NULL)),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE timer_lifecycle_reviews (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        review_id TEXT NOT NULL CHECK(length(review_id) BETWEEN 1 AND 128),
        rule_id TEXT NOT NULL CHECK(length(rule_id) BETWEEN 1 AND 128),
        occurrence_id TEXT NOT NULL CHECK(length(occurrence_id) BETWEEN 1 AND 128),
        occurrence_generation INTEGER NOT NULL CHECK(occurrence_generation >= 0),
        main_core_run_id INTEGER NOT NULL CHECK(main_core_run_id >= 1),
        expected_rule_version INTEGER NOT NULL CHECK(expected_rule_version >= 1),
        expected_activity_epoch INTEGER NOT NULL CHECK(expected_activity_epoch >= 0),
        evidence_json TEXT NOT NULL CHECK(length(evidence_json) BETWEEN 2 AND 64000),
        status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN (
            'PENDING', 'KEPT', 'COMPLETED', 'STALE', 'ERROR_KEEP', 'SKIPPED'
        )),
        decision TEXT NOT NULL DEFAULT '' CHECK(decision IN (
            '', 'KEEP_ONGOING', 'KEEP_UNCERTAIN',
            'COMPLETE_FULFILLED', 'COMPLETE_ENDED'
        )),
        error_code TEXT NOT NULL DEFAULT '' CHECK(length(error_code) <= 128),
        task_id INTEGER REFERENCES ai_tasks(task_id) ON DELETE SET NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        decided_at TEXT,
        applied_at TEXT,
        PRIMARY KEY(profile_id, instance_id, review_id),
        UNIQUE(profile_id, instance_id, rule_id, occurrence_id, occurrence_generation),
        FOREIGN KEY(profile_id, instance_id, rule_id)
            REFERENCES timer_rules(profile_id, instance_id, rule_id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, occurrence_id)
            REFERENCES timer_occurrences(profile_id, instance_id, occurrence_id)
            ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, main_core_run_id)
            REFERENCES instance_core_runs(profile_id, instance_id, run_id)
            ON DELETE CASCADE
    );

CREATE TABLE timer_occurrence_rolls (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        rule_id TEXT NOT NULL,
        last_materialized_due_at TEXT NOT NULL,
        through_at TEXT NOT NULL,
        latest_missed_due_at TEXT,
        coalesced_count INTEGER NOT NULL CHECK(coalesced_count >= 0),
        next_future_due_at TEXT,
        result_occurrence_ids_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(
            profile_id, instance_id, rule_id, last_materialized_due_at, through_at
        ),
        FOREIGN KEY(profile_id, instance_id, rule_id)
            REFERENCES timer_rules(profile_id, instance_id, rule_id) ON DELETE CASCADE
    );

CREATE TABLE timer_occurrences (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        occurrence_id TEXT NOT NULL CHECK(length(occurrence_id) BETWEEN 1 AND 128),
        stable_ref TEXT NOT NULL CHECK(length(stable_ref) BETWEEN 1 AND 128),
        rule_id TEXT NOT NULL CHECK(length(rule_id) BETWEEN 1 AND 128),
        original_due_at TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN (
            'SCHEDULED', 'WAITING', 'CLAIMED', 'RUNNING', 'WAITING_DELIVERY',
            'PAUSED', 'RECOVERING', 'COMPLETED', 'CANCELLED', 'FAILED',
            'MISSED_COALESCED'
        )),
        version INTEGER NOT NULL CHECK(version >= 1),
        generation INTEGER NOT NULL CHECK(generation >= 0),
        created_sequence INTEGER NOT NULL CHECK(created_sequence >= 1),
        created_at TEXT NOT NULL,
        execution_ref TEXT,
        delivery_ref TEXT,
        recovery_from TEXT CHECK(recovery_from IN ('CLAIMED', 'RUNNING', 'WAITING_DELIVERY')),
        last_operation_key TEXT NOT NULL DEFAULT '',
        last_operation_fingerprint TEXT NOT NULL DEFAULT '',
        PRIMARY KEY(profile_id, instance_id, occurrence_id),
        UNIQUE(profile_id, instance_id, stable_ref),
        UNIQUE(profile_id, instance_id, rule_id, original_due_at),
        UNIQUE(profile_id, instance_id, created_sequence),
        CHECK(last_operation_fingerprint = '' OR length(last_operation_fingerprint) = 64),
        CHECK((status = 'RECOVERING') = (recovery_from IS NOT NULL)),
        CHECK(status <> 'RUNNING' OR execution_ref IS NOT NULL),
        CHECK(status <> 'WAITING_DELIVERY' OR delivery_ref IS NOT NULL),
        FOREIGN KEY(profile_id, instance_id, rule_id)
            REFERENCES timer_rules(profile_id, instance_id, rule_id) ON DELETE CASCADE
    );

CREATE TABLE timer_operation_receipts (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL CHECK(length(idempotency_key) BETWEEN 1 AND 160),
        operation_kind TEXT NOT NULL CHECK(length(operation_kind) BETWEEN 1 AND 64),
        request_fingerprint TEXT NOT NULL CHECK(length(request_fingerprint) = 64),
        result_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id, idempotency_key),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE timer_rules (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        rule_id TEXT NOT NULL CHECK(length(rule_id) BETWEEN 1 AND 128),
        schedule_kind TEXT NOT NULL CHECK(schedule_kind IN (
            'ABSOLUTE', 'RELATIVE', 'WEEKLY', 'YEARLY'
        )),
        schedule_json TEXT NOT NULL,
        prompt TEXT NOT NULL CHECK(length(prompt) BETWEEN 1 AND 1000),
        fingerprint TEXT NOT NULL CHECK(length(fingerprint) = 64),
        status TEXT NOT NULL CHECK(status IN ('ACTIVE', 'PAUSED', 'CANCELLED', 'COMPLETED')),
        version INTEGER NOT NULL CHECK(version >= 1),
        created_sequence INTEGER NOT NULL CHECK(created_sequence >= 1),
        created_at TEXT NOT NULL,
        source_run_ref TEXT NOT NULL CHECK(length(source_run_ref) BETWEEN 1 AND 128),
        source_message_refs_json TEXT NOT NULL,
        last_operation_key TEXT NOT NULL DEFAULT '',
        last_operation_fingerprint TEXT NOT NULL DEFAULT '',
        PRIMARY KEY(profile_id, instance_id, rule_id),
        UNIQUE(profile_id, instance_id, created_sequence),
        CHECK(last_operation_fingerprint = '' OR length(last_operation_fingerprint) = 64),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE timer_run_refs (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        source_run_ref TEXT NOT NULL CHECK(length(source_run_ref) BETWEEN 1 AND 128),
        opaque_ref TEXT NOT NULL CHECK(length(opaque_ref) BETWEEN 8 AND 128),
        target TEXT NOT NULL CHECK(target IN ('SERIES', 'OCCURRENCE')),
        rule_id TEXT NOT NULL,
        occurrence_id TEXT,
        target_version INTEGER NOT NULL CHECK(target_version >= 1),
        created_at TEXT NOT NULL,
        PRIMARY KEY(profile_id, instance_id, source_run_ref, opaque_ref),
        UNIQUE(profile_id, instance_id, source_run_ref, target, rule_id, occurrence_id),
        CHECK(
            (target = 'SERIES' AND occurrence_id IS NULL)
            OR (target = 'OCCURRENCE' AND occurrence_id IS NOT NULL)
        ),
        FOREIGN KEY(profile_id, instance_id, rule_id)
            REFERENCES timer_rules(profile_id, instance_id, rule_id) ON DELETE CASCADE,
        FOREIGN KEY(profile_id, instance_id, occurrence_id)
            REFERENCES timer_occurrences(profile_id, instance_id, occurrence_id)
            ON DELETE CASCADE
    );

CREATE TABLE visual_observation_cache (
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
        contract_version INTEGER NOT NULL CHECK(contract_version >= 1),
        visible_facts TEXT NOT NULL
            CHECK(length(visible_facts) BETWEEN 1 AND 5000),
        ocr_text TEXT NOT NULL DEFAULT '' CHECK(length(ocr_text) <= 2000),
        subject_identity TEXT NOT NULL DEFAULT ''
            CHECK(length(subject_identity) <= 120),
        scene_description TEXT NOT NULL DEFAULT ''
            CHECK(length(scene_description) <= 5000),
        visual_style TEXT NOT NULL DEFAULT '' CHECK(length(visual_style) <= 120),
        sticker_type TEXT NOT NULL DEFAULT '' CHECK(length(sticker_type) <= 120),
        visible_text_state TEXT NOT NULL
            CHECK(visible_text_state IN ('HAS_VISIBLE_TEXT', 'NO_VISIBLE_TEXT')),
        safe INTEGER NOT NULL CHECK(safe IN (0, 1)),
        backend_id TEXT NOT NULL DEFAULT '' CHECK(length(backend_id) <= 200),
        model_id TEXT NOT NULL DEFAULT '' CHECK(length(model_id) <= 200),
        cached_at TEXT NOT NULL,
        last_used_at TEXT NOT NULL,
        PRIMARY KEY(
            profile_id,
            instance_id,
            sha256,
            contract_version
        ),
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE web_image_search_results (
        image_resource_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES web_search_sessions(session_id) ON DELETE CASCADE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        original_url TEXT NOT NULL DEFAULT '',
        thumbnail_url TEXT NOT NULL DEFAULT '',
        source_page_url TEXT NOT NULL DEFAULT '',
        source_domain TEXT NOT NULL DEFAULT '',
        title TEXT NOT NULL DEFAULT '',
        description TEXT NOT NULL DEFAULT '',
        provider_id TEXT NOT NULL DEFAULT '',
        provider_rank INTEGER NOT NULL DEFAULT 0 CHECK(provider_rank >= 0),
        cross_source_count INTEGER NOT NULL DEFAULT 1 CHECK(cross_source_count >= 1),
        width INTEGER CHECK(width IS NULL OR width >= 0),
        height INTEGER CHECK(height IS NULL OR height >= 0),
        mime_type TEXT NOT NULL DEFAULT '',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        retrieved_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        redacted_at TEXT,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE web_page_snapshots (
        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        resource_id TEXT NOT NULL UNIQUE
            REFERENCES web_search_results(resource_id) ON DELETE CASCADE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        content_hash TEXT NOT NULL DEFAULT '',
        token_estimate INTEGER NOT NULL DEFAULT 0 CHECK(token_estimate >= 0),
        status TEXT NOT NULL DEFAULT 'READ' CHECK(status IN (
            'NOT_READ', 'READ', 'FAILED', 'EXPIRED'
        )),
        error TEXT NOT NULL DEFAULT '',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        retrieved_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        redacted_at TEXT,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE web_search_providers (
        provider_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL REFERENCES role_profiles(profile_id) ON DELETE CASCADE,
        provider_kind TEXT NOT NULL CHECK(provider_kind IN (
            'TAVILY', 'BOCHA', 'BRAVE', 'FIRECRAWL', 'BAIDU_AI', 'EXA'
        )),
        display_name TEXT NOT NULL DEFAULT '',
        backend_id TEXT NOT NULL UNIQUE REFERENCES ai_backends(backend_id) ON DELETE RESTRICT,
        credential_id TEXT NOT NULL DEFAULT '',
        priority INTEGER NOT NULL DEFAULT 1 CHECK(priority >= 1),
        enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
        read_enabled INTEGER NOT NULL DEFAULT 0 CHECK(read_enabled IN (0, 1)),
        config_json TEXT NOT NULL DEFAULT '{}',
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        archived_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

CREATE TABLE web_search_results (
        resource_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES web_search_sessions(session_id) ON DELETE CASCADE,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        canonical_url TEXT NOT NULL DEFAULT '',
        domain TEXT NOT NULL DEFAULT '',
        snippet TEXT NOT NULL DEFAULT '',
        published_at TEXT,
        retrieved_at TEXT NOT NULL,
        provider_id TEXT NOT NULL DEFAULT '',
        provider_rank INTEGER NOT NULL DEFAULT 0 CHECK(provider_rank >= 0),
        cross_source_count INTEGER NOT NULL DEFAULT 1 CHECK(cross_source_count >= 1),
        read_status TEXT NOT NULL DEFAULT 'NOT_READ' CHECK(read_status IN (
            'NOT_READ', 'READ', 'FAILED', 'EXPIRED'
        )),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        expires_at TEXT NOT NULL,
        redacted_at TEXT,
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE web_search_sessions (
        session_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        caller_kind TEXT NOT NULL
            CHECK(caller_kind IN (
                'MAIN_CORE', 'BACKGROUND_AUTHOR', 'STICKER_COLLECTOR'
            )),
        caller_id TEXT NOT NULL DEFAULT '',
        core_run_id INTEGER,
        ai_task_id TEXT,
        purpose TEXT NOT NULL CHECK(purpose IN ('ANSWER_USER', 'SELF_EXPLORATION')),
        query TEXT NOT NULL,
        depth TEXT NOT NULL DEFAULT 'auto',
        freshness TEXT NOT NULL DEFAULT 'auto',
        status TEXT NOT NULL DEFAULT 'RUNNING' CHECK(status IN (
            'RUNNING', 'COMPLETED', 'PARTIAL', 'FAILED', 'CANCELLED', 'EXPIRED'
        )),
        deadline_at TEXT,
        partial_warning TEXT NOT NULL DEFAULT '',
        provider_count INTEGER NOT NULL DEFAULT 0 CHECK(provider_count >= 0),
        result_count INTEGER NOT NULL DEFAULT 0 CHECK(result_count >= 0),
        diagnostics_json TEXT NOT NULL DEFAULT '{}',
        error TEXT NOT NULL DEFAULT '',
        started_at TEXT NOT NULL,
        finished_at TEXT,
        expires_at TEXT NOT NULL,
        redacted_at TEXT, search_kind TEXT NOT NULL DEFAULT 'WEB'
        CHECK(search_kind IN ('WEB', 'IMAGE')), effective_caller_kind TEXT NOT NULL DEFAULT '',
        FOREIGN KEY(profile_id, instance_id)
            REFERENCES character_instances(profile_id, instance_id) ON DELETE CASCADE
    );

CREATE TABLE world_definitions (
        profile_id TEXT PRIMARY KEY
            REFERENCES role_profiles(profile_id) ON DELETE CASCADE,
        revision INTEGER NOT NULL DEFAULT 1 CHECK(revision >= 1),
        world_brief TEXT NOT NULL DEFAULT '',
        world_rules TEXT NOT NULL DEFAULT '',
        life_direction TEXT NOT NULL DEFAULT '',
        world_texture TEXT NOT NULL DEFAULT '',
        expansion_policy TEXT NOT NULL DEFAULT 'OPEN'
            CHECK(expansion_policy IN ('OPEN', 'CANON_GUARDED')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

CREATE TABLE world_lore_entries (
        lore_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL
            REFERENCES role_profiles(profile_id) ON DELETE CASCADE,
        revision INTEGER NOT NULL DEFAULT 1 CHECK(revision >= 1),
        title TEXT NOT NULL,
        aliases_json TEXT NOT NULL DEFAULT '[]',
        tags_json TEXT NOT NULL DEFAULT '[]',
        content TEXT NOT NULL,
        importance REAL NOT NULL DEFAULT 0.5
            CHECK(importance BETWEEN 0 AND 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(profile_id, title)
    );

CREATE UNIQUE INDEX idx_ai_api_models_active_key
        ON ai_api_models(package_id, model_key) WHERE archived_at IS NULL;

CREATE INDEX idx_ai_api_models_order
        ON ai_api_models(archived_at, priority, backend_id);

CREATE INDEX idx_ai_api_packages_profile ON ai_api_packages(profile_id, archived_at, display_name);

CREATE INDEX idx_ai_capability_pool_order
        ON ai_capability_pools(capability, enabled, priority DESC);

CREATE INDEX idx_ai_circuit_states_backend
        ON ai_circuit_states(backend_id, capability, updated_at DESC);

CREATE INDEX idx_ai_prompt_cache_state_probe
    ON ai_prompt_cache_capabilities(state, next_probe_at, probe_expires_at);

CREATE INDEX idx_ai_provider_attempts_invocation
            ON ai_provider_attempts(invocation_id, attempt_id);

CREATE INDEX idx_ai_provider_attempts_node
            ON ai_provider_attempts(node_id, round_no, attempt_no, attempt_id);

CREATE INDEX idx_ai_provider_attempts_sent
            ON ai_provider_attempts(sent_at, status, attempt_id);

CREATE INDEX idx_ai_task_attempts_task
        ON ai_task_attempts(task_id, attempt_id DESC);

CREATE INDEX idx_ai_task_audit_scope
        ON ai_task_audit(profile_id, instance_id, audit_id DESC);

CREATE INDEX idx_ai_task_audit_task
        ON ai_task_audit(task_id, audit_id DESC);

CREATE UNIQUE INDEX idx_ai_tasks_active_mutex
        ON ai_tasks(profile_id, instance_id, mutex_key)
        WHERE mutex_key IS NOT NULL AND status IN (
            'RUNNING', 'PAUSE_REQUESTED', 'CANCEL_REQUESTED'
        );

CREATE INDEX idx_ai_tasks_due
        ON ai_tasks(status, due_at, priority DESC, task_id);

CREATE UNIQUE INDEX idx_ai_tasks_idempotency
        ON ai_tasks(
            profile_id, instance_id, task_type,
            idempotency_key, generation
        ) WHERE idempotency_key IS NOT NULL;


CREATE INDEX idx_ai_tasks_scope
        ON ai_tasks(profile_id, instance_id, updated_at DESC);

CREATE INDEX idx_ai_work_events_workflow
            ON ai_work_events(workflow_id, sequence, event_id);

CREATE INDEX idx_ai_work_nodes_role_purpose
            ON ai_work_nodes(workflow_id, node_role, purpose, sequence);

CREATE INDEX idx_ai_work_nodes_workflow
            ON ai_work_nodes(workflow_id, sequence, node_id);

CREATE INDEX idx_ai_workflows_expiry
            ON ai_workflows(expires_at, workflow_id);

CREATE INDEX idx_ai_workflows_scope
            ON ai_workflows(profile_id, instance_id, started_at DESC, workflow_id DESC);

CREATE INDEX idx_background_author_due
    ON background_author_states(status, next_due_at, hard_due_at, profile_id, instance_id);

CREATE INDEX idx_background_foreground_lease
    ON background_instances(foreground_lease_until, profile_id, instance_id)
    WHERE foreground_lease_until IS NOT NULL;

CREATE INDEX idx_background_publications_recent
    ON background_author_publications(
        profile_id, instance_id, author_kind, publication_id DESC
    );

CREATE INDEX idx_background_story_sources_exposure
    ON background_story_sources(profile_id, instance_id, shown_count, story_source_id DESC);

CREATE INDEX idx_background_story_sources_instance
    ON background_story_sources(profile_id, instance_id, story_source_id DESC);

CREATE INDEX idx_background_timeline_recent
    ON background_role_timeline_events(
        profile_id, instance_id, frame_end_at DESC, event_id DESC
    );

CREATE INDEX idx_bg_event_story_by_story
    ON background_timeline_event_story_sources(story_source_id, event_id DESC);

CREATE INDEX idx_bg_story_run_exposures_story
    ON background_story_run_exposures(story_source_id, run_id);

CREATE INDEX idx_character_instances_profile_scope
        ON character_instances(profile_id, scope, updated_at DESC);

CREATE INDEX idx_character_intent_events
        ON character_intent_events(profile_id, instance_id, event_id DESC);

CREATE INDEX idx_character_intent_evidence
        ON character_intent_evidence(intent_id, revision, evidence_id);

CREATE INDEX idx_character_intents_active
        ON character_intents(profile_id, instance_id, status, priority DESC, updated_at);

CREATE UNIQUE INDEX idx_character_intents_active_conflict
        ON character_intents(profile_id, instance_id, conflict_key)
        WHERE conflict_key <> '' AND status IN ('OPEN', 'PLANNED', 'IN_PROGRESS', 'BLOCKED');

CREATE INDEX idx_character_model_revisions_fingerprint
        ON character_model_revisions(profile_id, content_fingerprint);

CREATE UNIQUE INDEX idx_contact_evidence_one_active_reservation
        ON contact_evidence_reservations(profile_id, instance_id,
            evidence_kind, evidence_ref)
        WHERE status = 'RESERVED';

CREATE INDEX idx_contact_state_due
        ON instance_contact_state(next_check_at, profile_id, instance_id);

CREATE INDEX idx_creative_boundaries_profile
    ON creative_boundaries(profile_id, enabled, severity);

CREATE INDEX idx_deferred_batches_due
        ON deferred_message_batches(status, due_at, profile_id, instance_id);

CREATE INDEX idx_dialogue_summaries_latest
        ON dialogue_summaries(profile_id, instance_id, version DESC);

CREATE INDEX idx_expression_batches_instance_status
    ON instance_expression_batches(profile_id, instance_id, status, created_at);

CREATE INDEX idx_expression_interruption_events_message
            ON expression_interruption_events(profile_id, instance_id, inbound_message_id);

CREATE INDEX idx_file_assets_instance
        ON file_assets(profile_id, instance_id, created_at DESC);

CREATE UNIQUE INDEX idx_file_assets_storage_path
        ON file_assets(storage_relpath);

CREATE INDEX idx_file_jobs_instance
        ON file_generation_jobs(profile_id, instance_id, created_at DESC);

CREATE INDEX idx_file_jobs_status
        ON file_generation_jobs(status, updated_at);

CREATE INDEX idx_group_flow_force_due
        ON group_flow_windows(status, quiet_due_at, dynamic_due_at, direct_due_at);

CREATE INDEX idx_group_flow_judge_due
        ON group_flow_windows(status, next_judge_at, window_id);

CREATE INDEX idx_group_flow_members_order
        ON group_flow_window_members(profile_id, instance_id, message_id);

CREATE UNIQUE INDEX idx_group_flow_one_collecting
        ON group_flow_windows(profile_id, instance_id) WHERE status = 'COLLECTING';

CREATE UNIQUE INDEX idx_group_flow_one_pipeline
        ON group_flow_windows(profile_id, instance_id)
        WHERE status IN ('JUDGING', 'READY', 'RUNNING', 'WAITING_FIRST_ATTEMPT');

CREATE INDEX idx_group_flow_ready
        ON group_flow_windows(status, ready_at, window_id);

CREATE INDEX idx_group_reply_relocation_due
        ON group_reply_relocation_states(candidate_recheck_at, window_id)
        WHERE candidate_recheck_at IS NOT NULL;

CREATE INDEX idx_important_todos_pending
        ON important_todos(profile_id, instance_id, status, available_at, created_at);

CREATE INDEX idx_inbound_recall_receipts_pending
        ON inbound_recall_receipts(status, expires_at, received_at);

CREATE INDEX idx_inbound_recall_states_due
        ON inbound_message_recall_states(status, grace_until, lease_until);

CREATE INDEX idx_inbound_voice_admissions_status
        ON inbound_voice_admissions(status, updated_at);

CREATE INDEX idx_instance_main_core_lease_expiry
        ON instance_main_core_occupancies(status, lease_expires_at, profile_id, instance_id);

CREATE UNIQUE INDEX idx_instance_messages_expression_ordinal
        ON instance_messages(profile_id, instance_id, expression_batch_id, expression_ordinal)
        WHERE expression_batch_id IS NOT NULL AND expression_ordinal IS NOT NULL;

CREATE UNIQUE INDEX idx_instance_messages_idempotency
        ON instance_messages(profile_id, instance_id, idempotency_key)
        WHERE idempotency_key IS NOT NULL;

CREATE INDEX idx_instance_messages_occurred
        ON instance_messages(profile_id, instance_id, occurred_at, message_id);

CREATE INDEX idx_instance_messages_recent
        ON instance_messages(profile_id, instance_id, message_id DESC);

CREATE INDEX idx_instance_outbox_dependency
        ON instance_outbox(profile_id, instance_id, depends_on_idempotency_key);

CREATE INDEX idx_instance_outbox_expression_due
        ON instance_outbox(status, not_before_at, expression_batch_id, expression_ordinal);

CREATE UNIQUE INDEX idx_instance_outbox_expression_ordinal
        ON instance_outbox(profile_id, instance_id, expression_batch_id, expression_ordinal)
        WHERE expression_batch_id IS NOT NULL AND expression_ordinal IS NOT NULL;

CREATE UNIQUE INDEX idx_instance_outbox_expression_step
        ON instance_outbox(profile_id, instance_id, expression_batch_id, expression_step_ordinal)
        WHERE expression_batch_id IS NOT NULL AND expression_step_ordinal IS NOT NULL;

CREATE INDEX idx_instance_outbox_origin
        ON instance_outbox(profile_id, instance_id, origin_kind, origin_task_id);

CREATE INDEX idx_instance_outbox_status
        ON instance_outbox(status, created_at);

CREATE UNIQUE INDEX idx_instance_runs_owner
        ON instance_core_runs(profile_id, instance_id, run_id);

CREATE INDEX idx_instance_runs_recent
        ON instance_core_runs(profile_id, instance_id, run_id DESC);

CREATE INDEX idx_instance_wakeups_due ON instance_wakeups(status, due_at);

CREATE UNIQUE INDEX idx_instance_wakeups_idempotency
        ON instance_wakeups(profile_id, instance_id, idempotency_key)
        WHERE idempotency_key IS NOT NULL;

CREATE INDEX idx_knowledge_audit_instance
        ON knowledge_audit(profile_id, instance_id, audit_id DESC);

CREATE INDEX idx_knowledge_batches_instance
        ON knowledge_batches(profile_id, instance_id, batch_id DESC);

CREATE INDEX idx_knowledge_batches_task ON knowledge_batches(ai_task_id);

CREATE INDEX idx_knowledge_fact_revisions_current
        ON knowledge_fact_revisions(knowledge_fact_id, revision DESC);

CREATE INDEX idx_knowledge_fact_terms_lookup
        ON knowledge_fact_terms(normalized_term, term_kind, knowledge_fact_id);

CREATE INDEX idx_knowledge_marks_batch ON knowledge_message_marks(batch_id);

CREATE INDEX idx_main_core_work_file_pending
        ON main_core_work_file_bindings(
            status, profile_id, instance_id, work_ref, request_ref
        );

CREATE INDEX idx_main_core_work_lease_expiry
        ON main_core_work_checkpoints(
            status, lease_expires_at, profile_id, instance_id, work_ref
        );

CREATE INDEX idx_main_core_work_recovery_ready
        ON main_core_work_checkpoints(
            status, created_at, profile_id, instance_id, work_ref
        );

CREATE INDEX idx_main_core_work_recovery_wake_ready
        ON main_core_work_recovery_wakes(
            status, created_at, profile_id, instance_id, work_ref
        );

CREATE INDEX idx_media_assets_cleanup
        ON media_assets(file_status, expires_at);

CREATE INDEX idx_media_assets_hash
        ON media_assets(profile_id, instance_id, sha256);

CREATE INDEX idx_media_assets_instance_recent
        ON media_assets(profile_id, instance_id, created_at DESC);

CREATE INDEX idx_media_assets_task ON media_assets(ai_task_id);

CREATE INDEX idx_media_cleanup_instance
        ON media_cleanup_events(profile_id, instance_id, cleanup_id DESC);

CREATE INDEX idx_media_message_links_message
        ON media_asset_message_links(profile_id, instance_id, message_id, ordinal);

CREATE INDEX idx_media_projections_asset
        ON media_projections(asset_id, version DESC);

CREATE INDEX idx_media_retention_holds_active
        ON media_retention_holds(asset_id, released_at);

CREATE INDEX idx_memory_revision_sources_message
        ON memory_revision_sources(profile_id, instance_id, message_id)
        WHERE source_kind = 'MESSAGE';

CREATE INDEX idx_memory_revisions_current
        ON memory_revisions(memory_id, revision DESC);

CREATE INDEX idx_memory_terms_lookup ON memory_terms(normalized_term, memory_id);

CREATE INDEX idx_message_fragments_ledger
        ON instance_message_fragments(
            profile_id, instance_id, ledger_message_id, fragment_ordinal
        );

CREATE INDEX idx_message_fragments_retractable
        ON instance_message_fragments(
            profile_id, instance_id, self_retraction_supported,
            retraction_status, retractable_until
        );

CREATE INDEX idx_message_retraction_fragment_attempts_status
    ON message_retraction_fragment_attempts(action_id, status);

CREATE INDEX idx_platform_media_reference
        ON platform_message_media_refs(
            profile_id, instance_id, platform_message_id, ordinal
        );

CREATE INDEX idx_player_profile_entries_scope
        ON player_profile_entries(profile_id, instance_id, subject_key, entry_id);

CREATE INDEX idx_player_profile_revisions_history
        ON player_profile_entry_revisions(
            profile_id, instance_id, subject_key, entry_id, entry_version DESC
        );

CREATE INDEX idx_recall_documents_scope
        ON recall_documents(profile_id, instance_id, authority_status, source_type);

CREATE INDEX idx_recall_documents_time
        ON recall_documents(profile_id, instance_id, occurred_at, valid_from, valid_until);

CREATE INDEX idx_recall_edges_source
        ON recall_edges(profile_id, instance_id, source_document_key, status);

CREATE INDEX idx_recall_edges_target
        ON recall_edges(profile_id, instance_id, target_document_key, status);

CREATE INDEX idx_recall_embeddings_generation
        ON recall_embeddings(generation_id, document_key);

CREATE UNIQUE INDEX idx_recall_generation_active
        ON recall_index_generations(profile_id, instance_id)
        WHERE active = 1;

CREATE INDEX idx_recall_generation_build
        ON recall_index_generations(status, updated_at, generation_id);

CREATE INDEX idx_recall_graph_edges_source
        ON recall_graph_edges(profile_id, instance_id, source_node_key, status);

CREATE INDEX idx_recall_graph_edges_target
        ON recall_graph_edges(profile_id, instance_id, target_node_key, status);

CREATE INDEX idx_recall_graph_nodes_document
        ON recall_graph_nodes(document_key, node_type);

CREATE INDEX idx_recall_graph_nodes_scope
        ON recall_graph_nodes(profile_id, instance_id, node_type, stable_ref);

CREATE INDEX idx_recall_outbox_ready
        ON recall_index_outbox(status, not_before, task_id);

CREATE INDEX idx_recall_outbox_scope
        ON recall_index_outbox(profile_id, instance_id, status, task_id);

CREATE INDEX idx_recall_probe_reports_recent
        ON recall_probe_reports(profile_id, instance_id, report_id DESC);

CREATE INDEX idx_recall_scene_members_document
        ON recall_scene_members(document_key, scene_key);

CREATE INDEX idx_recall_scenes_scope
        ON recall_scenes(profile_id, instance_id, scene_level, occurred_from);

CREATE INDEX idx_retraction_actions_batch
        ON message_retraction_actions(
            profile_id, instance_id, expression_batch_id, step_ordinal
        );

CREATE INDEX idx_retraction_actions_due
        ON message_retraction_actions(status, not_before_at, action_id);

CREATE INDEX idx_retraction_actions_target
        ON message_retraction_actions(profile_id, instance_id, target_message_ref);

CREATE INDEX idx_runtime_file_cleanup_scope
        ON runtime_file_cleanup_queue(
            not_before_at, attempt_count, cleanup_id
        );

CREATE INDEX idx_send_permits_account
        ON platform_send_permits(platform_instance_id, account_key, status, expires_at);

CREATE INDEX idx_send_permits_group
        ON platform_send_permits(platform_instance_id, target_id, status, expires_at);

CREATE INDEX idx_soulcore_logs_profile_category_recent
        ON soulcore_logs(profile_id, category, log_id DESC);

CREATE INDEX idx_soulcore_logs_profile_instance_recent
        ON soulcore_logs(profile_id, instance_id, log_id DESC);

CREATE INDEX idx_soulcore_logs_profile_level_recent
        ON soulcore_logs(profile_id, level, log_id DESC);

CREATE INDEX idx_soulcore_logs_profile_recent
        ON soulcore_logs(profile_id, log_id DESC);

CREATE INDEX idx_sticker_candidates_instance
        ON sticker_candidates(profile_id, instance_id, status, created_at DESC);

CREATE INDEX idx_sticker_candidates_library
        ON sticker_candidates(target_library_id, status, created_at DESC);

CREATE INDEX idx_sticker_candidates_retry
        ON sticker_candidates(status, recoverable, next_retry_at);

CREATE INDEX idx_sticker_checks_candidate
        ON sticker_check_revisions(candidate_id, revision DESC);

CREATE INDEX idx_sticker_fingerprints_scope
        ON sticker_fingerprints(profile_id, instance_id, visual_group);

CREATE INDEX idx_sticker_imports_instance
        ON sticker_import_events(profile_id, instance_id, created_at DESC);

CREATE INDEX idx_sticker_instance_item_states_item
    ON sticker_instance_item_states(item_id);

CREATE INDEX idx_sticker_intake_entries_session
        ON sticker_intake_entries(session_id, status, created_at, entry_id);

CREATE INDEX idx_sticker_intake_sessions_expiry
        ON sticker_intake_sessions(status, expires_at);

CREATE INDEX idx_sticker_intake_sessions_scope
        ON sticker_intake_sessions(profile_id, scope, created_at DESC);

CREATE INDEX idx_sticker_items_cluster
        ON sticker_items(cluster_id, status, source_kind);

CREATE INDEX idx_sticker_items_instance
        ON sticker_items(profile_id, instance_id, status, last_used_at, created_at DESC);

CREATE INDEX idx_sticker_items_library
        ON sticker_items(library_id, status, usage_type, last_used_at, created_at DESC);

CREATE INDEX idx_sticker_items_search
        ON sticker_items(profile_id, instance_id, status, search_index);

CREATE INDEX idx_sticker_reinforcements_item
        ON sticker_reinforcements(profile_id, instance_id, item_id, created_at DESC);

CREATE INDEX idx_sticker_run_candidates_run
        ON sticker_run_candidates(profile_id, instance_id, run_id, expires_at);

CREATE INDEX idx_sticker_trigger_due
        ON sticker_trigger_states(profile_id, cooldown_until, updated_at);

CREATE INDEX idx_sticker_usages_instance
        ON sticker_usages(profile_id, instance_id, created_at DESC);

CREATE INDEX idx_timer_lifecycle_reviews_pending
        ON timer_lifecycle_reviews(status, updated_at, profile_id, instance_id, review_id);

CREATE INDEX idx_timer_lifecycle_reviews_rule_recent
        ON timer_lifecycle_reviews(
            profile_id, instance_id, rule_id, created_at DESC, review_id
        );

CREATE INDEX idx_timer_occurrences_claim_order
        ON timer_occurrences(
            profile_id, instance_id, status, original_due_at, created_sequence, stable_ref
        );

CREATE INDEX idx_timer_occurrences_rule_due
        ON timer_occurrences(profile_id, instance_id, rule_id, original_due_at);

CREATE INDEX idx_timer_occurrences_status_due
        ON timer_occurrences(
            status, original_due_at, profile_id, instance_id, created_sequence, stable_ref
        );

CREATE INDEX idx_timer_rules_scope_status_sequence
        ON timer_rules(profile_id, instance_id, status, created_sequence, rule_id);

CREATE INDEX idx_turn_buffer_classification
        ON conversation_turn_buffer_batches(status, updated_at, batch_id);

CREATE INDEX idx_turn_buffer_due
        ON conversation_turn_buffer_batches(status, due_at, batch_id);

CREATE INDEX idx_turn_buffer_members_instance_message
        ON conversation_turn_buffer_members(profile_id, instance_id, message_id);

CREATE UNIQUE INDEX idx_turn_buffer_one_active_instance
        ON conversation_turn_buffer_batches(profile_id, instance_id)
        WHERE status IN ('PENDING', 'CLASSIFYING', 'WAITING', 'CLAIMED');

CREATE INDEX idx_visual_observation_cache_lru
    ON visual_observation_cache(
        contract_version,
        last_used_at,
        profile_id,
        instance_id,
        sha256
    );

CREATE INDEX idx_web_image_results_scope
        ON web_image_search_results(profile_id, instance_id, expires_at);

CREATE INDEX idx_web_image_results_session
        ON web_image_search_results(session_id, provider_rank, image_resource_id);

CREATE INDEX idx_web_page_snapshots_expiry
        ON web_page_snapshots(expires_at) WHERE redacted_at IS NULL;

CREATE UNIQUE INDEX idx_web_search_provider_active_priority
        ON web_search_providers(profile_id, priority) WHERE archived_at IS NULL;

CREATE INDEX idx_web_search_provider_profile
        ON web_search_providers(profile_id, archived_at, enabled, priority);

CREATE INDEX idx_web_search_results_scope
        ON web_search_results(profile_id, instance_id, expires_at);

CREATE INDEX idx_web_search_results_session
        ON web_search_results(session_id, provider_rank, resource_id);

CREATE INDEX idx_web_search_sessions_expiry
        ON web_search_sessions(expires_at) WHERE redacted_at IS NULL;

CREATE INDEX idx_web_search_sessions_instance
        ON web_search_sessions(profile_id, instance_id, started_at DESC);

CREATE INDEX idx_world_lore_lookup
    ON world_lore_entries(profile_id, importance DESC, updated_at DESC);

CREATE UNIQUE INDEX uq_instance_main_core_active_occupancy
        ON instance_main_core_occupancies(profile_id, instance_id)
        WHERE status = 'ACTIVE';

CREATE UNIQUE INDEX uq_message_retraction_output_target
    ON message_retraction_actions(profile_id, instance_id, source_run_id, target_output_ordinal)
    WHERE target_output_ordinal IS NOT NULL;

CREATE UNIQUE INDEX uq_message_retraction_target_ref
    ON message_retraction_actions(profile_id, instance_id, target_message_ref)
    WHERE target_message_ref IS NOT NULL;

CREATE UNIQUE INDEX uq_sticker_intake_candidate
        ON sticker_intake_entries(candidate_id)
        WHERE candidate_id IS NOT NULL;

CREATE UNIQUE INDEX uq_sticker_intake_content
        ON sticker_intake_entries(session_id, content_sha256)
        WHERE content_sha256 <> '';

CREATE UNIQUE INDEX uq_sticker_intake_scope_active
        ON sticker_intake_sessions(profile_id, scope)
        WHERE status IN ('UPLOADING', 'RUNNING', 'REVIEW', 'FINALIZING');

CREATE UNIQUE INDEX uq_sticker_libraries_core
        ON sticker_libraries(profile_id, scope)
        WHERE library_kind = 'CORE';

CREATE UNIQUE INDEX uq_sticker_libraries_private
        ON sticker_libraries(profile_id, instance_id)
        WHERE library_kind = 'PRIVATE';

CREATE UNIQUE INDEX uq_sticker_usage_outbox_expression
        ON sticker_usages(
            profile_id, instance_id, outbox_id, expression_ordinal, sticker_ref
        )
        WHERE outbox_id IS NOT NULL AND expression_ordinal IS NOT NULL;

CREATE UNIQUE INDEX uq_timer_rules_non_cancelled_fingerprint
        ON timer_rules(profile_id, instance_id, fingerprint)
        WHERE status NOT IN ('CANCELLED', 'COMPLETED');

CREATE TRIGGER trg_instance_messages_expression_owner_insert
    BEFORE INSERT ON instance_messages
    WHEN NEW.expression_batch_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM instance_expression_batches batch
        WHERE batch.batch_id = NEW.expression_batch_id
          AND batch.profile_id = NEW.profile_id
          AND batch.instance_id = NEW.instance_id
    )
    BEGIN
        SELECT RAISE(ABORT, 'expression batch belongs to another instance');
    END;

CREATE TRIGGER trg_instance_messages_expression_owner_update
    BEFORE UPDATE OF expression_batch_id, profile_id, instance_id ON instance_messages
    WHEN NEW.expression_batch_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM instance_expression_batches batch
        WHERE batch.batch_id = NEW.expression_batch_id
          AND batch.profile_id = NEW.profile_id
          AND batch.instance_id = NEW.instance_id
    )
    BEGIN
        SELECT RAISE(ABORT, 'expression batch belongs to another instance');
    END;

CREATE TRIGGER trg_instance_outbox_expression_owner_insert
    BEFORE INSERT ON instance_outbox
    WHEN NEW.expression_batch_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM instance_expression_batches batch
        WHERE batch.batch_id = NEW.expression_batch_id
          AND batch.profile_id = NEW.profile_id
          AND batch.instance_id = NEW.instance_id
    )
    BEGIN
        SELECT RAISE(ABORT, 'expression batch belongs to another instance');
    END;

CREATE TRIGGER trg_instance_outbox_expression_owner_update
    BEFORE UPDATE OF expression_batch_id, profile_id, instance_id ON instance_outbox
    WHEN NEW.expression_batch_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM instance_expression_batches batch
        WHERE batch.batch_id = NEW.expression_batch_id
          AND batch.profile_id = NEW.profile_id
          AND batch.instance_id = NEW.instance_id
    )
    BEGIN
        SELECT RAISE(ABORT, 'expression batch belongs to another instance');
    END;

CREATE TRIGGER trg_outbox_existing_retraction_order_insert
    BEFORE INSERT ON instance_outbox
    WHEN NEW.expression_batch_id IS NOT NULL
     AND NEW.expression_step_ordinal IS NOT NULL
     AND EXISTS (
        SELECT 1 FROM message_retraction_actions action
        WHERE action.profile_id = NEW.profile_id
          AND action.instance_id = NEW.instance_id
          AND action.expression_batch_id = NEW.expression_batch_id
          AND action.target_message_ref IS NOT NULL
          AND action.step_ordinal >= NEW.expression_step_ordinal
    )
    BEGIN
        SELECT RAISE(ABORT, 'visible output must follow existing-message retractions');
    END;

CREATE TRIGGER trg_outbox_retraction_step_collision_insert
    BEFORE INSERT ON instance_outbox
    WHEN NEW.expression_batch_id IS NOT NULL
     AND NEW.expression_step_ordinal IS NOT NULL
     AND EXISTS (
        SELECT 1 FROM message_retraction_actions action
        WHERE action.profile_id = NEW.profile_id
          AND action.instance_id = NEW.instance_id
          AND action.expression_batch_id = NEW.expression_batch_id
          AND action.step_ordinal = NEW.expression_step_ordinal
    )
    BEGIN
        SELECT RAISE(ABORT, 'expression step ordinal is already used by a retraction action');
    END;

CREATE TRIGGER trg_recall_memory_entry_delete
BEFORE DELETE ON memories
BEGIN
    INSERT OR IGNORE INTO recall_index_outbox(
        task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, not_before, created_at, updated_at
    ) VALUES (
        'memory:' || OLD.memory_id || ':delete:' || OLD.current_revision,
        OLD.profile_id, OLD.instance_id, 'MEMORY', CAST(OLD.memory_id AS TEXT), 'DELETE',
        OLD.current_revision, 'PENDING', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
    );
END;

CREATE TRIGGER trg_recall_memory_entry_update
AFTER UPDATE OF status, current_revision ON memories
BEGIN
    INSERT OR REPLACE INTO recall_index_outbox(
        task_id, task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, lease_owner, lease_token, lease_until, attempt_count,
        last_error, not_before, created_at, updated_at
    ) VALUES (
        (SELECT task_id FROM recall_index_outbox WHERE task_key =
            'memory:' || NEW.memory_id || ':entry:' || NEW.current_revision),
        'memory:' || NEW.memory_id || ':entry:' || NEW.current_revision,
        NEW.profile_id, NEW.instance_id, 'MEMORY', CAST(NEW.memory_id AS TEXT), 'UPSERT',
        NEW.current_revision, 'PENDING', NULL, 0, NULL, 0, '',
        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
    );
END;

CREATE TRIGGER trg_recall_memory_revision_insert
AFTER INSERT ON memory_revisions
BEGIN
    INSERT OR IGNORE INTO recall_index_outbox(
        task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, not_before, created_at, updated_at
    )
    SELECT 'memory:' || NEW.memory_id || ':' || NEW.revision || ':upsert',
           profile_id, instance_id, 'MEMORY', CAST(NEW.memory_id AS TEXT), 'UPSERT',
           NEW.revision, 'PENDING', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
    FROM memories WHERE memory_id = NEW.memory_id;
END;

CREATE TRIGGER trg_recall_message_delete
BEFORE DELETE ON instance_messages
BEGIN
    INSERT OR IGNORE INTO recall_index_outbox(
        task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, not_before, created_at, updated_at
    ) VALUES (
        'message:' || OLD.message_id || ':delete', OLD.profile_id, OLD.instance_id,
        'MESSAGE', CAST(OLD.message_id AS TEXT), 'DELETE', 1, 'PENDING',
        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
    );
END;

CREATE TRIGGER trg_recall_message_insert
AFTER INSERT ON instance_messages
BEGIN
    INSERT OR IGNORE INTO recall_index_outbox(
        task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, not_before, created_at, updated_at
    ) VALUES (
        'message:' || NEW.message_id || ':upsert', NEW.profile_id, NEW.instance_id,
        'MESSAGE', CAST(NEW.message_id AS TEXT), 'UPSERT', 1, 'PENDING',
        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
    );
END;

CREATE TRIGGER trg_recall_message_visibility_update
AFTER UPDATE OF delivery_status, plain_text, knowledge_eligibility ON instance_messages
BEGIN
    INSERT OR REPLACE INTO recall_index_outbox(
        task_id, task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, lease_owner, lease_token, lease_until, attempt_count,
        last_error, not_before, created_at, updated_at
    ) VALUES (
        (SELECT task_id FROM recall_index_outbox WHERE task_key =
            'message:' || NEW.message_id || ':update'),
        'message:' || NEW.message_id || ':update', NEW.profile_id, NEW.instance_id,
        'MESSAGE', CAST(NEW.message_id AS TEXT), 'UPSERT', 1, 'PENDING', NULL, 0, NULL,
        0, '', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
    );
END;

CREATE TRIGGER trg_recall_role_current_delete
BEFORE DELETE ON background_role_current_views
BEGIN
    INSERT OR IGNORE INTO recall_index_outbox(
        task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, not_before, created_at, updated_at
    ) VALUES (
        'role-current:' || OLD.profile_id || ':' || OLD.instance_id || ':delete:' || OLD.revision,
        OLD.profile_id, OLD.instance_id, 'ROLE_CURRENT', OLD.instance_id, 'DELETE',
        OLD.revision, 'PENDING', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
    );
END;

CREATE TRIGGER trg_recall_role_current_insert
AFTER INSERT ON background_role_current_views
BEGIN
    INSERT OR IGNORE INTO recall_index_outbox(
        task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, not_before, created_at, updated_at
    ) VALUES (
        'role-current:' || NEW.profile_id || ':' || NEW.instance_id || ':' || NEW.revision,
        NEW.profile_id, NEW.instance_id, 'ROLE_CURRENT', NEW.instance_id, 'UPSERT',
        NEW.revision, 'PENDING', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
    );
END;

CREATE TRIGGER trg_recall_role_current_update
AFTER UPDATE ON background_role_current_views
BEGIN
    INSERT OR IGNORE INTO recall_index_outbox(
        task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, not_before, created_at, updated_at
    ) VALUES (
        'role-current:' || NEW.profile_id || ':' || NEW.instance_id || ':' || NEW.revision,
        NEW.profile_id, NEW.instance_id, 'ROLE_CURRENT', NEW.instance_id, 'UPSERT',
        NEW.revision, 'PENDING', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
    );
END;

CREATE TRIGGER trg_recall_role_event_delete
BEFORE DELETE ON background_role_timeline_events
BEGIN
    INSERT OR IGNORE INTO recall_index_outbox(
        task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, not_before, created_at, updated_at
    ) VALUES (
        'role-event:' || OLD.event_id || ':delete', OLD.profile_id, OLD.instance_id,
        'ROLE_EVENT', CAST(OLD.event_id AS TEXT), 'DELETE', 1, 'PENDING',
        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
    );
END;

CREATE TRIGGER trg_recall_role_event_insert
AFTER INSERT ON background_role_timeline_events
BEGIN
    INSERT OR IGNORE INTO recall_index_outbox(
        task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, not_before, created_at, updated_at
    ) VALUES (
        'role-event:' || NEW.event_id || ':upsert', NEW.profile_id, NEW.instance_id,
        'ROLE_EVENT', CAST(NEW.event_id AS TEXT), 'UPSERT', 1, 'PENDING',
        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
    );
END;

CREATE TRIGGER trg_recall_summary_delete
BEFORE DELETE ON dialogue_summaries
BEGIN
    INSERT OR IGNORE INTO recall_index_outbox(
        task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, not_before, created_at, updated_at
    ) VALUES (
        'summary:' || OLD.summary_id || ':delete:' || OLD.version,
        OLD.profile_id, OLD.instance_id, 'DIALOGUE_SUMMARY', CAST(OLD.summary_id AS TEXT),
        'DELETE', OLD.version, 'PENDING', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
        CURRENT_TIMESTAMP
    );
END;

CREATE TRIGGER trg_recall_summary_insert
AFTER INSERT ON dialogue_summaries
BEGIN
    INSERT OR IGNORE INTO recall_index_outbox(
        task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, not_before, created_at, updated_at
    ) VALUES (
        'summary:' || NEW.summary_id || ':upsert:' || NEW.version,
        NEW.profile_id, NEW.instance_id, 'DIALOGUE_SUMMARY', CAST(NEW.summary_id AS TEXT),
        'UPSERT', NEW.version, 'PENDING', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
        CURRENT_TIMESTAMP
    );
END;

CREATE TRIGGER trg_recall_world_entry_delete
BEFORE DELETE ON knowledge_fact_entries
BEGIN
    INSERT OR IGNORE INTO recall_index_outbox(
        task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, not_before, created_at, updated_at
    ) VALUES (
        'world:' || OLD.knowledge_fact_id || ':delete:' || OLD.current_revision,
        OLD.profile_id, OLD.instance_id, 'WORLD_INFO', CAST(OLD.knowledge_fact_id AS TEXT),
        'DELETE', OLD.current_revision, 'PENDING', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
        CURRENT_TIMESTAMP
    );
END;

CREATE TRIGGER trg_recall_world_entry_update
AFTER UPDATE OF status, current_revision ON knowledge_fact_entries
BEGIN
    INSERT OR REPLACE INTO recall_index_outbox(
        task_id, task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, lease_owner, lease_token, lease_until, attempt_count,
        last_error, not_before, created_at, updated_at
    ) VALUES (
        (SELECT task_id FROM recall_index_outbox WHERE task_key =
            'world:' || NEW.knowledge_fact_id || ':entry:' || NEW.current_revision),
        'world:' || NEW.knowledge_fact_id || ':entry:' || NEW.current_revision,
        NEW.profile_id, NEW.instance_id, 'WORLD_INFO', CAST(NEW.knowledge_fact_id AS TEXT),
        'UPSERT', NEW.current_revision, 'PENDING', NULL, 0, NULL, 0, '',
        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
    );
END;

CREATE TRIGGER trg_recall_world_revision_insert
AFTER INSERT ON knowledge_fact_revisions
BEGIN
    INSERT OR IGNORE INTO recall_index_outbox(
        task_key, profile_id, instance_id, source_type, source_key, operation,
        source_version, status, not_before, created_at, updated_at
    )
    SELECT 'world:' || NEW.knowledge_fact_id || ':' || NEW.revision || ':upsert',
           profile_id, instance_id, 'WORLD_INFO', CAST(NEW.knowledge_fact_id AS TEXT),
           'UPSERT', NEW.revision, 'PENDING', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
           CURRENT_TIMESTAMP
    FROM knowledge_fact_entries WHERE knowledge_fact_id = NEW.knowledge_fact_id;
END;

CREATE TRIGGER trg_retraction_action_count_delete
    AFTER DELETE ON message_retraction_actions
    BEGIN
        UPDATE instance_expression_batches
        SET retraction_count = retraction_count - 1
        WHERE profile_id = OLD.profile_id
          AND instance_id = OLD.instance_id
          AND batch_id = OLD.expression_batch_id;
    END;

CREATE TRIGGER trg_retraction_action_count_insert
    AFTER INSERT ON message_retraction_actions
    BEGIN
        UPDATE instance_expression_batches
        SET retraction_count = retraction_count + 1, updated_at = NEW.updated_at
        WHERE profile_id = NEW.profile_id
          AND instance_id = NEW.instance_id
          AND batch_id = NEW.expression_batch_id;
    END;

CREATE TRIGGER trg_retraction_existing_target_order_insert
    BEFORE INSERT ON message_retraction_actions
    WHEN NEW.target_message_ref IS NOT NULL AND EXISTS (
        SELECT 1 FROM instance_outbox item
        WHERE item.profile_id = NEW.profile_id
          AND item.instance_id = NEW.instance_id
          AND item.expression_batch_id = NEW.expression_batch_id
          AND item.expression_step_ordinal IS NOT NULL
          AND item.expression_step_ordinal <= NEW.step_ordinal
    )
    BEGIN
        SELECT RAISE(ABORT, 'existing-message retraction must precede visible outputs');
    END;
""".replace("__DEFAULT_STICKER_REQUIREMENTS__", DEFAULT_STICKER_REQUIREMENTS).replace(
    "__INSTANCE_CHAT_POLICIES_SQL__", INSTANCE_CHAT_POLICIES_SQL
)

FINGERPRINT = hashlib.sha256(SQL.encode("utf-8")).hexdigest()

__all__ = ["FINGERPRINT", "INSTANCE_CHAT_POLICIES_SQL", "SQL"]
