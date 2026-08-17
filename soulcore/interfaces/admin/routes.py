"""Frozen Plugin Page route contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PageRoute:
    suffix: str
    handler: str
    methods: tuple[str, ...]
    description: str


PAGE_ROUTES = (
    PageRoute(
        "console_bootstrap",
        "console_bootstrap",
        ("GET",),
        "Read the human-oriented SoulCore console bootstrap",
    ),
    PageRoute(
        "player_bootstrap",
        "player_bootstrap",
        ("GET",),
        "Read the player-facing SoulCore shell bootstrap",
    ),
    PageRoute(
        "player_guide_acknowledge",
        "player_guide_acknowledge",
        ("POST",),
        "Remember the current player guide version as seen",
    ),
    PageRoute(
        "advanced_guide_acknowledge",
        "advanced_guide_acknowledge",
        ("POST",),
        "Remember the current advanced settings guide version as seen",
    ),
    PageRoute("player_now", "player_now", ("GET",), "Read the character's current life"),
    PageRoute("player_contacts", "player_contacts", ("GET",), "Read player-facing contacts"),
    PageRoute(
        "player_relationship",
        "player_relationship",
        ("GET",),
        "Read one player-facing relationship",
    ),
    PageRoute("player_about", "player_about", ("GET",), "Read player-facing role details"),
    PageRoute(
        "role_package_export_prepare",
        "role_package_export_prepare",
        ("POST",),
        "Prepare one portable package for the selected existing role",
    ),
    PageRoute(
        "role_package_download",
        "role_package_download",
        ("GET",),
        "Download one short-lived verified role package",
    ),
    PageRoute(
        "role_package_import_upload",
        "role_package_import_upload",
        ("POST",),
        "Upload, validate, preview, or abort one portable role package",
    ),
    PageRoute(
        "role_package_import_apply",
        "role_package_import_apply",
        ("POST",),
        "Atomically apply one locked role-package preview",
    ),
    PageRoute("release_notes", "release_notes", ("GET",), "Read player-facing release notes"),
    PageRoute(
        "quick_setup_status",
        "quick_setup_status",
        ("GET",),
        "Read whether the current role still needs the first-run setup choice",
    ),
    PageRoute(
        "quick_setup_snapshot",
        "quick_setup_snapshot",
        ("GET",),
        "Read the current role's guided model setup state",
    ),
    PageRoute(
        "quick_setup_configure",
        "quick_setup_configure",
        ("POST",),
        "Probe and apply one guided model setup step",
    ),
    PageRoute(
        "quick_setup_web_configure",
        "quick_setup_web_configure",
        ("POST",),
        "Probe and apply the guided web-search setup step",
    ),
    PageRoute(
        "quick_setup_sticker_configure",
        "quick_setup_sticker_configure",
        ("POST",),
        "Apply one guided sticker choice to private and group chat",
    ),
    PageRoute(
        "quick_setup_life_configure",
        "quick_setup_life_configure",
        ("POST",),
        "Apply one guided role-wide life choice to existing and future chats",
    ),
    PageRoute(
        "quick_setup_contact_configure",
        "quick_setup_contact_configure",
        ("POST",),
        "Apply one guided contact habit to private and group defaults",
    ),
    PageRoute(
        "quick_setup_character_generate",
        "quick_setup_character_generate",
        ("POST",),
        "Generate an unsaved public character-profile draft from the selected AstrBot Persona",
    ),
    PageRoute(
        "quick_setup_decision",
        "quick_setup_decision",
        ("POST",),
        "Record that the current role will be configured manually",
    ),
    PageRoute(
        "quick_setup_finish",
        "quick_setup_finish",
        ("POST",),
        "Finish guided setup after revalidating the main model assignment",
    ),
    PageRoute(
        "schema_recovery",
        "schema_recovery",
        ("GET",),
        "Read a bounded database recovery choice after startup refusal",
    ),
    PageRoute(
        "schema_recovery_action",
        "schema_recovery_action",
        ("POST",),
        "Execute one explicitly confirmed database recovery choice",
    ),
    PageRoute(
        "settings_snapshot",
        "settings_snapshot",
        ("GET",),
        "Read all SoulCore settings for one profile and scope",
    ),
    PageRoute(
        "settings_section",
        "settings_section",
        ("POST",),
        "Autosave one bounded SoulCore settings section",
    ),
    PageRoute(
        "identity_annotations",
        "identity_annotations",
        ("POST",),
        "Preview or apply confirmed semantic identity annotations",
    ),
    PageRoute(
        "instance_workspace",
        "instance_workspace",
        ("GET",),
        "Read one human-oriented conversation workspace",
    ),
    PageRoute(
        "background_workspace",
        "background_workspace",
        ("GET",),
        "Read the current role state, five authors, story modules and lived timeline",
    ),
    PageRoute(
        "background_action",
        "background_action",
        ("POST",),
        "Apply one world-seed edit or versioned background enable, schedule, wake or reset action",
    ),
    PageRoute(
        "player_profile_snapshot",
        "player_profile_snapshot",
        ("GET",),
        "Read the human-oriented player profile workspace",
    ),
    PageRoute(
        "delivery_failure_acknowledge",
        "delivery_failure_acknowledge",
        ("POST",),
        "Acknowledge one terminal delivery failure without retrying or deleting it",
    ),
    PageRoute(
        "player_profile_entry",
        "player_profile_entry",
        ("GET",),
        "Read one player profile entry and its revision history",
    ),
    PageRoute(
        "player_profile_action",
        "player_profile_action",
        ("POST",),
        "Create, update, withdraw, restore or permanently delete one profile entry",
    ),
    PageRoute(
        "ai_work_records",
        "ai_work_records",
        ("GET",),
        "List causal SoulCore AI work records",
    ),
    PageRoute(
        "ai_work_record",
        "ai_work_record",
        ("GET",),
        "Read one causal SoulCore AI work record",
    ),
    PageRoute(
        "ai_work_attempt_debug",
        "ai_work_attempt_debug",
        ("GET",),
        "Read one credential-safe SoulCore model attempt diagnostic",
    ),
    PageRoute(
        "ai_work_attempt_raw",
        "ai_work_attempt_raw",
        ("GET",),
        "Read one credential-safe provider exchange",
    ),
    PageRoute(
        "console_profile", "console_profile", ("POST",), "Remember the selected console profile"
    ),
    PageRoute(
        "instance_contact_override",
        "instance_contact_override",
        ("GET",),
        "Read one chat's SoulCore, image and Contact policies for the settings workspace",
    ),
    PageRoute(
        "platform_contact_policy",
        "platform_contact_policy",
        ("GET",),
        "Read one platform connection policy for the settings workspace",
    ),
    PageRoute("instances", "instances", ("GET",), "List conversation role instances"),
    PageRoute("support_bundle", "support_bundle", ("GET",), "Export SoulCore support snapshot"),
    PageRoute("context_summary", "context_summary", ("POST",), "Run a dialogue summary task"),
    PageRoute("context_dry_run", "context_dry_run", ("POST",), "Build a context budget report"),
    PageRoute(
        "knowledge_snapshot",
        "knowledge_snapshot",
        ("GET",),
        "Read instance memories and named knowledge facts",
    ),
    PageRoute(
        "knowledge_form",
        "knowledge_form",
        ("POST",),
        "Run the background knowledge formation plugin",
    ),
    PageRoute("recall_probe", "recall_probe", ("POST",), "Run a unified read-only recall probe"),
    PageRoute(
        "recall_configuration",
        "recall_configuration",
        ("GET",),
        "Read effective Recall providers and index readiness",
    ),
    PageRoute(
        "recall_configuration_update",
        "recall_configuration_update",
        ("POST",),
        "Update role Recall provider inheritance",
    ),
    PageRoute("recall_rebuild", "recall_rebuild", ("POST",), "Queue a Recall index rebuild"),
    PageRoute(
        "recall_integrity",
        "recall_integrity",
        ("POST",),
        "Run a Recall index integrity check",
    ),
    PageRoute(
        "recall_benchmark",
        "recall_benchmark",
        ("POST",),
        "Run a bounded live Recall ranking benchmark",
    ),
    PageRoute(
        "knowledge_record",
        "knowledge_record",
        ("POST",),
        "Create, update, disable, restore or delete one knowledge record",
    ),
    PageRoute(
        "character_intent_action",
        "character_intent_action",
        ("POST",),
        "Manage one character intent",
    ),
    PageRoute(
        "image_snapshot", "image_snapshot", ("GET",), "Read one instance media asset snapshot"
    ),
    PageRoute(
        "image_preview",
        "image_preview",
        ("GET",),
        "Read one bounded WebP preview for an owned image asset",
    ),
    PageRoute(
        "image_download",
        "image_download",
        ("GET",),
        "Download one verified owned image asset",
    ),
    PageRoute(
        "file_artifacts", "file_artifacts", ("GET",), "Read one instance file artifact snapshot"
    ),
    PageRoute(
        "file_artifact_action",
        "file_artifact_action",
        ("POST",),
        "Manage one generated file artifact",
    ),
    PageRoute("ai_api_packages", "ai_api_packages", ("GET",), "Read API configuration packages"),
    PageRoute("ai_api_package", "ai_api_package", ("POST",), "Save one API configuration package"),
    PageRoute(
        "ai_api_package_credential",
        "ai_api_package_credential",
        ("POST",),
        "Set one API package credential",
    ),
    PageRoute("ai_api_model", "ai_api_model", ("POST",), "Save one model in an API package"),
    PageRoute(
        "ai_api_package_probe",
        "ai_api_package_probe",
        ("POST",),
        "Probe one API package connection",
    ),
    PageRoute(
        "ai_api_model_probe", "ai_api_model_probe", ("POST",), "Probe one configured API model"
    ),
    PageRoute(
        "web_provider", "web_provider", ("POST",), "Create or update one web research provider"
    ),
    PageRoute(
        "web_provider_credential",
        "web_provider_credential",
        ("POST",),
        "Set one web research provider credential",
    ),
    PageRoute(
        "web_provider_probe", "web_provider_probe", ("POST",), "Probe one web research provider"
    ),
    PageRoute("web_snapshot", "web_snapshot", ("GET",), "Read paginated web research diagnostics"),
    PageRoute(
        "api/stickers/snapshot", "sticker_snapshot", ("GET",), "Read one instance sticker library"
    ),
    PageRoute(
        "timer_lifecycle_snapshot",
        "timer_lifecycle_snapshot",
        ("GET",),
        "Read one instance Timer lifecycle review summary",
    ),
    PageRoute(
        "api/stickers/action", "sticker_action", ("POST",), "Manage a sticker candidate or item"
    ),
    PageRoute("api/stickers/run", "sticker_run", ("POST",), "Run the sticker collector"),
    PageRoute("api/stickers/stop", "sticker_stop", ("POST",), "Stop the active sticker collector"),
    PageRoute(
        "api/stickers/intake",
        "sticker_intake",
        ("GET",),
        "Read the current or selected sticker intake batch",
    ),
    PageRoute(
        "api/stickers/intake/start",
        "sticker_intake_start",
        ("POST",),
        "Start one upload or search sticker intake batch",
    ),
    PageRoute(
        "api/stickers/intake/upload",
        "sticker_intake_upload",
        ("POST",),
        "Upload one image into a sticker intake batch",
    ),
    PageRoute(
        "api/stickers/intake/action",
        "sticker_intake_action",
        ("POST",),
        "Review, finish, retry, or cancel a sticker intake batch",
    ),
    PageRoute(
        "api/stickers/intake/preview",
        "sticker_intake_preview",
        ("GET",),
        "Read one bounded WebP sticker intake preview",
    ),
    PageRoute(
        "api/stickers/references",
        "sticker_references",
        ("POST",),
        "Manage character identity references",
    ),
    PageRoute(
        "reset_instance",
        "reset_instance",
        ("POST",),
        "Reset one conversation and start initialization",
    ),
)
