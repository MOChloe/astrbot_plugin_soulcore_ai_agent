from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ....storage.sqlite.dialogue_turns import context_eligible_sql
from .support import _dump, _load

SummaryCoverage = tuple[tuple[int, int, int], ...]


def model_visible_summary_coverage(
    values: Sequence[Sequence[int]],
) -> SummaryCoverage:
    rows = {
        (int(value[0]), int(value[1]), int(value[2]))
        for value in values
        if len(value) == 3
        and int(value[0]) > 0
        and int(value[1]) > 0
        and int(value[2]) >= int(value[1])
    }
    return tuple(sorted(rows))


def summary_coverage_objects(
    values: Sequence[tuple[int, int, int]],
) -> list[dict[str, int]]:
    return [
        {
            "summary_id": summary_id,
            "covered_from_message_id": covered_from,
            "covered_through_message_id": covered_through,
        }
        for summary_id, covered_from, covered_through in values
    ]


def project_model_visible_message_ids(
    conn: Any,
    *,
    run_id: int,
    node_id: int,
    visible_ids: tuple[int, ...],
    visible_summary_ids: set[int],
    coverage: SummaryCoverage,
) -> dict[str, Any] | None:
    row = _projection_target(conn, run_id=run_id, node_id=node_id)
    if row is None or not _visible_messages_belong_to_target(conn, row, visible_ids):
        return None
    covered_message_ids = _covered_message_ids(conn, row, coverage)
    if covered_message_ids is None:
        return None
    projected_ids = set(visible_ids) | covered_message_ids
    projected_summary_ids = set(visible_summary_ids)
    run_request = _merge_projection_payload(
        row["request_json"],
        metadata_key="metadata",
        projected_ids=projected_ids,
        projected_summary_ids=projected_summary_ids,
        coverage=coverage,
    )
    node_input = _merge_projection_payload(
        row["input_json"],
        metadata_key=None,
        projected_ids=projected_ids,
        projected_summary_ids=projected_summary_ids,
        coverage=coverage,
    )
    if not _store_projection_payloads(
        conn,
        run_id=run_id,
        node_id=node_id,
        run_request=run_request,
        node_input=node_input,
    ):
        return None
    return {
        "model_visible_message_ids": sorted(projected_ids),
        "model_visible_summary_ids": sorted(projected_summary_ids),
        "model_visible_summary_coverage": summary_coverage_objects(coverage),
    }


def _projection_target(conn: Any, *, run_id: int, node_id: int) -> Any:
    return conn.execute(
        """SELECT run.request_json, node.input_json,
            run.profile_id, run.instance_id
        FROM instance_core_runs run
        JOIN ai_workflows workflow
          ON workflow.workflow_id = run.workflow_id
        JOIN ai_work_nodes node
          ON node.workflow_id = workflow.workflow_id
        WHERE run.run_id = ? AND node.node_id = ?
          AND run.status = 'RUNNING' AND workflow.status = 'RUNNING'
          AND node.status = 'RUNNING'""",
        (int(run_id), int(node_id)),
    ).fetchone()


def _visible_messages_belong_to_target(
    conn: Any,
    row: Mapping[str, Any],
    visible_ids: tuple[int, ...],
) -> bool:
    if not visible_ids:
        return True
    placeholders = ",".join("?" for _ in visible_ids)
    visible = conn.execute(
        f"""SELECT COUNT(*) AS amount FROM instance_messages
        WHERE profile_id = ? AND instance_id = ?
          AND message_id IN ({placeholders})
          AND {context_eligible_sql("instance_messages")}""",
        (row["profile_id"], row["instance_id"], *visible_ids),
    ).fetchone()
    return visible is not None and int(visible["amount"] or 0) == len(visible_ids)


def _covered_message_ids(
    conn: Any,
    row: Mapping[str, Any],
    coverage: SummaryCoverage,
) -> set[int] | None:
    covered_message_ids: set[int] = set()
    for summary_id, covered_from, covered_through in coverage:
        summary = conn.execute(
            """SELECT 1 FROM dialogue_summaries
            WHERE summary_id = ? AND profile_id = ? AND instance_id = ?
              AND covered_from_message_id = ?
              AND covered_through_message_id = ?""",
            (
                summary_id,
                row["profile_id"],
                row["instance_id"],
                covered_from,
                covered_through,
            ),
        ).fetchone()
        if summary is None:
            return None
        covered_message_ids.update(
            int(message["message_id"])
            for message in conn.execute(
                """SELECT message_id FROM instance_messages
                WHERE profile_id = ? AND instance_id = ?
                  AND message_id BETWEEN ? AND ?""",
                (
                    row["profile_id"],
                    row["instance_id"],
                    covered_from,
                    covered_through,
                ),
            )
        )
    return covered_message_ids


def _merge_projection_payload(
    serialized: Any,
    *,
    metadata_key: str | None,
    projected_ids: set[int],
    projected_summary_ids: set[int],
    coverage: SummaryCoverage,
) -> dict[str, Any]:
    loaded = _load(serialized) if serialized else {}
    payload = dict(loaded) if isinstance(loaded, Mapping) else {}
    nested = payload.get(metadata_key) if metadata_key is not None else payload
    target = dict(nested) if isinstance(nested, Mapping) else {}
    target["model_visible_message_ids"] = sorted(
        _positive_ints(target.get("model_visible_message_ids")) | projected_ids
    )
    target["model_visible_summary_ids"] = sorted(
        _positive_ints(target.get("model_visible_summary_ids")) | projected_summary_ids
    )
    target["model_visible_summary_coverage"] = _merge_summary_coverage(
        target.get("model_visible_summary_coverage"), coverage
    )
    if metadata_key is not None:
        payload[metadata_key] = target
    else:
        payload = target
    return payload


def _positive_ints(value: Any) -> set[int]:
    return {int(item) for item in value or () if int(item) > 0}


def _merge_summary_coverage(
    existing: Any,
    current: SummaryCoverage,
) -> list[dict[str, int]]:
    merged = set(_stored_summary_coverage(existing))
    merged.update(current)
    return summary_coverage_objects(sorted(merged))


def _stored_summary_coverage(value: Any) -> SummaryCoverage:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    rows: list[tuple[int, int, int]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        summary_id = int(item.get("summary_id") or 0)
        covered_from = int(item.get("covered_from_message_id") or 0)
        covered_through = int(item.get("covered_through_message_id") or 0)
        if summary_id > 0 and covered_from > 0 and covered_through >= covered_from:
            rows.append((summary_id, covered_from, covered_through))
    return tuple(rows)


def _store_projection_payloads(
    conn: Any,
    *,
    run_id: int,
    node_id: int,
    run_request: Mapping[str, Any],
    node_input: Mapping[str, Any],
) -> bool:
    run_changed = conn.execute(
        """UPDATE instance_core_runs SET request_json = ?
        WHERE run_id = ? AND status = 'RUNNING'""",
        (_dump(run_request), int(run_id)),
    ).rowcount
    node_changed = conn.execute(
        """UPDATE ai_work_nodes SET input_json = ?
        WHERE node_id = ? AND status = 'RUNNING'""",
        (_dump(node_input), int(node_id)),
    ).rowcount
    return run_changed == 1 and node_changed == 1
