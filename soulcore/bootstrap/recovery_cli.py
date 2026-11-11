"""Offline recovery: python -m soulcore.bootstrap.recovery_cli --help."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..storage.sqlite.backup import SQLiteBackupManager
from ..storage.sqlite.recovery_publish import recover_interrupted_restore
from ..storage.sqlite.recovery_restore import restore_source
from ..storage.sqlite.recovery_sources import list_sources
from .runtime_fence import RuntimeOwnershipFence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SoulCore 离线备份检查与还原（还原前停止 AstrBot）"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="实际的 astrbot_plugin_soulcore 数据目录"
    )
    parser.add_argument("action", choices=("list", "restore"))
    parser.add_argument("--source", default="", help="list 输出中的备份编号")
    parser.add_argument("--confirm", default="", help="还原时填写：还原备份")
    args = parser.parse_args(argv)
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        parser.error("数据目录不存在；不会自动创建一个空目录")
    manager = SQLiteBackupManager(
        data_dir / "soulcore.sqlite3",
        data_dir / "backups" / "soulcore.sqlite3",
        file_artifact_root=data_dir / "file_artifacts",
    )
    try:
        if args.action == "list":
            sources = list_sources(manager)
            print(f"备份目录：{manager.backup_path.parent}")
            for source in sources:
                view = source.public_view()
                suffix = "（已标记过期，只允许明确选择后还原）" if source.stale else ""
                print(
                    f"{source.id}  {view['created_at']}  {source.label}{suffix}\n  {source.path.name}"
                )
            if not sources:
                print(
                    "未找到候选备份。另查旧目录 data/soulcore_backups/astrbot_plugin_soulcore/ 及服务器快照。"
                )
            print("列表仅表示发现候选文件；还原时会校验数据库、完整性、文件制品和恢复包哈希。")
            return 0
        if args.confirm != "还原备份":
            parser.error('还原必须显式提供 --confirm "还原备份"；会覆盖当前状态，恢复到备份时刻')
        owner = RuntimeOwnershipFence.acquire(data_dir)
        try:
            with manager.write_fence.hold():
                recover_interrupted_restore(manager.database_path)
                source = next(
                    (item for item in list_sources(manager) if item.id == args.source), None
                )
                if source is None:
                    raise ValueError("未找到该备份编号，请重新执行 list")
                archive = restore_source(manager, source)
            print("校验并还原完成。现在可以启动 AstrBot。")
            if archive:
                print(f"还原前现场：{archive.path}\nSHA256：{archive.sha256}")
            return 0
        finally:
            owner.close()
    except Exception as exc:
        print(f"还原未完成：{type(exc).__name__}: {exc}")
        print("请保留原数据、恢复包和临时还原目录；不要清空或反复重建。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
