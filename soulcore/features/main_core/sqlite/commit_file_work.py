from __future__ import annotations

from datetime import timedelta

from ..work_checkpoint import (
    WorkCallbackLease,
    WorkRecoveryAction,
    WorkRecoveryBaseline,
    WorkScope,
)
from ..work_checkpoint_repository import FreezeWorkCheckpointCommand
from ..work_file_runtime import FileWorkBindingSpec
from ..work_recovery import export_work_checkpoint
from .support import Any, _parse, sqlite3


class FileWorkCheckpointMixin:
    context: Any

    def _freeze_file_work(
        self,
        conn: sqlite3.Connection,
        created_file_jobs: list[tuple[str, dict[str, Any]]],
    ) -> None:
        from .work_checkpoint_operations import WorkCheckpointSqliteOperations
        from .work_file_operations import WorkFileSqliteOperations

        if not created_file_jobs:
            if self.context.work_checkpoint_snapshot is not None:
                raise ValueError("a file work checkpoint requires created file jobs")
            return
        snapshot = self.context.work_checkpoint_snapshot
        if snapshot is None:
            raise ValueError("file generation requires an explicit Main Core work snapshot")
        if snapshot.status != "ACTIVE":
            raise ValueError("file generation requires one active Main Core work snapshot")
        if snapshot.work_ref != f"work:{self.context.run_id}:1":
            raise ValueError("file work must belong to the committing Main Core run")
        slot_by_request = _file_slot_bindings(snapshot, created_file_jobs)
        created_at = _parse(self.context.now)
        if created_at is None:
            raise ValueError("file work requires an aware commit timestamp")
        checkpoint = export_work_checkpoint(
            snapshot,
            scope=WorkScope(self.context.profile_id, self.context.instance_id),
            checkpoint_version=1,
            run_generation=1,
            callback_sequence=0,
            baseline=WorkRecoveryBaseline(
                self.context.expected_activity_epoch,
                self.context.expected_state_epoch,
                _file_permission_revision(conn, self.context),
                0,
            ),
            allowed_actions=(
                WorkRecoveryAction.REASSESS_PLAN,
                WorkRecoveryAction.UPDATE_WORK,
                WorkRecoveryAction.COMPLETE_WORK,
                WorkRecoveryAction.CANCEL_WORK,
            ),
            lease=WorkCallbackLease(
                owner=f"file-work:{self.context.run_id}",
                token=self.context.run_id,
                expires_at=created_at + timedelta(days=6, hours=23),
            ),
            created_at=created_at,
            expires_at=created_at + timedelta(days=7),
            controlled_resource_refs=self.context.work_controlled_resource_refs,
        )
        WorkCheckpointSqliteOperations().freeze_checkpoint(
            conn,
            FreezeWorkCheckpointCommand(
                checkpoint,
                f"freeze:file-work:{self.context.run_id}",
                created_at,
            ),
        )
        WorkFileSqliteOperations().bind_file_jobs(
            conn,
            checkpoint,
            tuple(
                FileWorkBindingSpec(
                    job_id,
                    str(request["request_ref"]),
                    slot_by_request[str(request["request_ref"])],
                )
                for job_id, request in created_file_jobs
            ),
            created_at,
        )


def _file_slot_bindings(snapshot: Any, jobs: list[tuple[str, dict[str, Any]]]) -> dict[str, str]:
    request_refs = {str(request.get("request_ref") or "") for _, request in jobs}
    slot_by_request: dict[str, str] = {}
    for slot in snapshot.result_slots:
        for resource_ref in slot.resource_refs:
            if resource_ref not in request_refs:
                continue
            if resource_ref in slot_by_request:
                raise ValueError("each file request must bind exactly one work result slot")
            slot_by_request[resource_ref] = slot.slot_id
    if set(slot_by_request) != request_refs or "" in request_refs:
        raise ValueError("every committed file request requires a work result slot")
    return slot_by_request


def _file_permission_revision(conn: sqlite3.Connection, context: Any) -> int:
    row = conn.execute(
        """SELECT rp.file_artifacts_enabled
        FROM role_profiles rp
        JOIN character_instances ci ON ci.profile_id = rp.profile_id
        WHERE rp.profile_id = ? AND ci.instance_id = ?""",
        (context.profile_id, context.instance_id),
    ).fetchone()
    if row is None:
        return 0
    return int(bool(row[0]))


__all__ = ["FileWorkCheckpointMixin"]
