"""Durable file-artifact task execution."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from ...contracts.ai_models import AIWorkPurpose
from ...shared.prompt_document import (
    TrustedPromptMarkup,
    join_prompt_markup,
    project_prompt_text,
    prompt_markup_block,
    prompt_markup_record,
)
from ..ai import record_structured_acceptance, record_structured_rejection
from ..profiles.ports import ProfilesRepositoryPort
from .artifacts import FileArtifactService, PDFImageAsset
from .lifecycle import FILE_ARTIFACTS_DISABLED_REASON, FileArtifactsDisabled
from .ports import FileMediaRepositoryPort, FileRepositoryPort

_FACTUAL_POLICY = "基于事实材料"
_CREATIVE_POLICY = "允许目标内创作"
_LAYOUT_STYLES = {
    "自动安排": "AUTO",
    "清晰正式": "REPORT",
    "杂志式图文": "EDITORIAL",
    "数据概览": "DATA_BRIEF",
}
_PLANNING_TASK_DEFINITION = "你先在心里做一轮编辑梳理，这步不产出成品，只留笔记给自己下一步写作用。"
_AUTHOR_TASK_DEFINITION = (
    "根据委托和你刚才的笔记，写出读者可以直接阅读的成品。"
    "你可以重写、合并、拆分段落，补充解释和过渡，但笔记不是新事实来源。"
)
_PLANNING_OUTPUT_CONTRACT = (
    "只记下委托本身尚未直接决定的东西：结构怎么搭、材料怎么取舍、顺序怎么排。"
    "简短、形式随意，不要复述委托内容。"
)
_PLANNING_IMAGE_OUTPUT_CONTRACT = (
    "只记下委托本身尚未直接决定的东西：结构怎么搭、材料怎么取舍、顺序怎么排，"
    "以及配图怎么取舍、放在哪里。简短、形式随意，不要复述委托内容。"
)


class FileModelPort(Protocol):
    async def generate_text(self, **values: object) -> Any: ...


class VisualAssetResolver(Protocol):
    async def asset_file_path(self, **values: object) -> Any: ...


@dataclass(frozen=True)
class _FilePromptContext:
    identity_context: Any
    identity_catalog: Any
    identity_scope: str
    planning_task_definition: str
    author_task_definition: str
    delegation: TrustedPromptMarkup
    image_references: tuple[str, ...]


def file_artifact_operation_timeout_seconds(file_format: str) -> int:
    return 3000 if str(file_format or "").strip().upper() == "PDF" else 300


class FileArtifactTaskExecutor:
    def __init__(
        self,
        *,
        file_repository: FileRepositoryPort,
        profiles_repository: ProfilesRepositoryPort,
        media_repository: FileMediaRepositoryPort,
        model_gateway: FileModelPort,
        service: FileArtifactService,
        visual_service: VisualAssetResolver,
        identity: Any,
    ) -> None:
        self.files = file_repository
        self.profiles = profiles_repository
        self.media = media_repository
        self.model_gateway = model_gateway
        self.service = service
        self.visual_service = visual_service
        self.identity = identity

    async def execute(self, task: dict[str, Any], control: Any) -> dict[str, Any]:
        await control.check_control()
        job = await self.files.get_file_generation_job_for_task(int(task["task_id"]))
        if job is None:
            raise RuntimeError("file generation job is missing")
        if str(job.get("status") or "") == "SUCCEEDED" and job.get("todo_id"):
            return self._recovered(job)
        profile_id, instance_id = (
            str(task.get("profile_id") or ""),
            str(task.get("instance_id") or ""),
        )
        await self._require_enabled_or_pause(profile_id, control)
        route_umo = await self._route(profile_id, instance_id)
        if not await self.files.mark_file_generation_job_running(
            int(task["task_id"]), lease_token=int(task["lease_token"])
        ):
            await self._require_enabled_or_pause(profile_id, control)
            raise RuntimeError("file generation job lease is stale")
        payload = dict(task.get("input") or {})
        file_format = self.service.normalize_format(payload.get("file_format") or "")
        layout_preference = str(payload["layout_preference"])
        style = self.service.normalize_document_style(_LAYOUT_STYLES[layout_preference])
        images, descriptions = await self._document_images(
            profile_id, instance_id, file_format, payload
        )
        body = await self._generate_body(
            task,
            profile_id,
            instance_id,
            route_umo,
            file_format,
            layout_preference,
            payload,
            descriptions,
        )
        await self._require_enabled_or_pause(profile_id, control)
        return await self._publish(
            task, control, job, profile_id, instance_id, file_format, style, payload, body, images
        )

    async def _route(self, profile_id: str, instance_id: str) -> str:
        instance = await self.profiles.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise RuntimeError("file generation instance is missing")
        route_umo = str(instance.route_umo or "").strip()
        if not route_umo:
            raise RuntimeError("file generation instance route is unavailable")
        return route_umo

    async def _document_images(
        self,
        profile_id: str,
        instance_id: str,
        file_format: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, PDFImageAsset], list[str]]:
        asset_ids = list(
            dict.fromkeys(
                str(item or "").strip()
                for item in payload.get("image_asset_ids") or []
                if str(item or "").strip()
            )
        )
        if len(asset_ids) > 5:
            raise RuntimeError("file generation image selection exceeds the limit")
        if asset_ids and file_format != "PDF":
            raise RuntimeError("only PDF generation may resolve document images")
        images, lines = {}, []
        for ordinal, asset_id in enumerate(asset_ids, start=1):
            image, description = await self._document_image(
                profile_id, instance_id, asset_id, ordinal
            )
            reference = f"I{ordinal}"
            images[reference] = image
            lines.append(f"- {reference}：{description}")
        return images, lines

    async def _document_image(
        self, profile_id: str, instance_id: str, asset_id: str, ordinal: int
    ) -> tuple[PDFImageAsset, str]:
        asset = await self.media.get_media_asset(
            asset_id, profile_id=profile_id, instance_id=instance_id
        )
        if asset is None or not str(asset.mime_type).startswith("image/"):
            raise RuntimeError("controlled document image is unavailable")
        if int(asset.width or 0) * int(asset.height or 0) > 100_000_000:
            raise RuntimeError("controlled document image exceeds the pixel budget")
        path = await self.visual_service.asset_file_path(
            profile_id=profile_id, instance_id=instance_id, asset_id=asset_id
        )
        if path is None:
            raise RuntimeError("controlled document image file is unavailable")
        projection = await self.media.get_latest_media_projection(asset_id)
        description = self._projection_description(projection)
        if not description:
            description = f"受控图片{ordinal}（{asset.width or '?'}×{asset.height or '?'}）"
        description = description[:500]
        return PDFImageAsset(path=path, description=description), description

    async def _generate_body(
        self,
        task: dict[str, Any],
        profile_id: str,
        instance_id: str,
        route_umo: str,
        file_format: str,
        layout_preference: str,
        payload: dict[str, Any],
        image_lines: list[str],
    ) -> str:
        del route_umo
        timeout = file_artifact_operation_timeout_seconds(file_format)
        prompt_context = await self._prepare_prompt_context(
            profile_id,
            instance_id,
            file_format,
            layout_preference,
            payload,
            image_lines,
        )
        planning_notes = await self._generate_plan(
            task,
            profile_id,
            instance_id,
            file_format,
            timeout,
            prompt_context,
        )
        return await self._generate_final_body(
            task,
            profile_id,
            instance_id,
            file_format,
            timeout,
            prompt_context,
            planning_notes,
        )

    async def _prepare_prompt_context(
        self,
        profile_id: str,
        instance_id: str,
        file_format: str,
        layout_preference: str,
        payload: dict[str, Any],
        image_lines: list[str],
    ) -> _FilePromptContext:
        identity_context, identity_catalog = await self.identity.catalog(profile_id, instance_id)
        identity_scope = str(identity_context.scope)
        projected_payload = self.identity.project_data_for_model(
            payload,
            identity_catalog,
            scope=identity_scope,
        )
        projected_images = [
            self.identity.project_for_model(value, identity_catalog, scope=identity_scope)
            for value in image_lines
        ]
        planning_task_definition = self.identity.project_for_model(
            _PLANNING_TASK_DEFINITION, identity_catalog, scope=identity_scope
        )
        author_task_definition = self.identity.project_for_model(
            _AUTHOR_TASK_DEFINITION, identity_catalog, scope=identity_scope
        )
        document_prompt = self._prompt(
            file_format,
            layout_preference,
            projected_payload,
            projected_images,
        )
        identity_tokens = self._referenced_identity_tokens(document_prompt, identity_catalog)
        delegation_blocks: list[TrustedPromptMarkup] = []
        if identity_tokens:
            delegation_blocks.append(
                prompt_markup_record(
                    "身份引用",
                    {
                        "目录": self._file_identity_prompt(
                            identity_catalog,
                            identity_tokens,
                        )
                    },
                )
            )
        delegation_blocks.append(document_prompt)
        delegation = project_prompt_text(
            join_prompt_markup(delegation_blocks),
            lambda value: self.identity.project_for_model(
                value,
                identity_catalog,
                scope=identity_scope,
            ),
        )
        return _FilePromptContext(
            identity_context=identity_context,
            identity_catalog=identity_catalog,
            identity_scope=identity_scope,
            planning_task_definition=planning_task_definition,
            author_task_definition=author_task_definition,
            delegation=delegation,
            image_references=tuple(
                reference
                for value in projected_images
                if (reference := self._image_prompt_values(value)[0])
            ),
        )

    @staticmethod
    def _referenced_identity_tokens(
        document_prompt: TrustedPromptMarkup,
        identity_catalog: Any,
    ) -> tuple[str, ...]:
        document = str(document_prompt)
        return tuple(
            token
            for token in identity_catalog.token_to_placeholder
            if str(token) and str(token) in document
        )

    @staticmethod
    def _file_identity_prompt(
        identity_catalog: Any,
        tokens: Sequence[str],
    ) -> str:
        rows = []
        for token in tokens:
            reference = str(identity_catalog.token_to_reference.get(token) or "").strip()
            label = str(identity_catalog.token_to_label.get(token) or "").strip()
            prefix = f"{reference} / " if reference else ""
            rows.append(f"{prefix}{token}：{label}")
        return (
            "下列标记只用于理解委托材料中的现实人物。成品直接使用适合读者的人类可读称呼，"
            "不得原样输出人物引用或身份标记。\n\n" + "\n".join(rows)
        )

    async def _generate_plan(
        self,
        task: dict[str, Any],
        profile_id: str,
        instance_id: str,
        file_format: str,
        timeout: int,
        prompt_context: _FilePromptContext,
    ) -> str:
        planning_contract = self.identity.project_for_model(
            self._planning_output_contract(prompt_context.image_references),
            prompt_context.identity_catalog,
            scope=prompt_context.identity_scope,
        )
        planning = await self.model_gateway.generate_text(
            task_definition=prompt_context.planning_task_definition,
            task_input=prompt_context.delegation,
            output_contract=planning_contract,
            profile_id=profile_id,
            instance_id=instance_id,
            capability="text.completion",
            owner_kind="file_artifact",
            owner_id=str(task["task_id"]),
            idempotency_key=f"file-artifact:{task['task_id']}:plan",
            work_purpose=AIWorkPurpose.FILE_GENERATION,
            logical_stage_key=f"file-artifact:{task['task_id']}:plan",
            operation_timeout_seconds=timeout,
        )
        planning_notes = str(planning.text or "").strip()
        if not planning_notes:
            await record_structured_rejection(
                model_gateway=self.model_gateway,
                completion=planning,
                round_no=1,
                error="文档作者没有完成私下梳理",
                terminal=True,
            )
            raise ValueError("文件正文模型没有完成私下梳理")
        await record_structured_acceptance(
            model_gateway=self.model_gateway,
            completion=planning,
            round_no=1,
            value={"planning_completed": True, "format": file_format},
        )
        return planning_notes

    @staticmethod
    def _planning_output_contract(image_references: Sequence[str]) -> str:
        if not image_references:
            return _PLANNING_OUTPUT_CONTRACT
        return _PLANNING_IMAGE_OUTPUT_CONTRACT

    async def _generate_final_body(
        self,
        task: dict[str, Any],
        profile_id: str,
        instance_id: str,
        file_format: str,
        timeout: int,
        prompt_context: _FilePromptContext,
        planning_notes: str,
    ) -> str:
        projected_plan = self.identity.project_for_model(
            planning_notes,
            prompt_context.identity_catalog,
            scope=prompt_context.identity_scope,
        )
        writing_input = join_prompt_markup(
            (
                prompt_context.delegation,
                prompt_markup_block("写作笔记", projected_plan),
            ),
        )
        output_contract = self.identity.project_for_model(
            self._body_output_contract(file_format, prompt_context.image_references),
            prompt_context.identity_catalog,
            scope=prompt_context.identity_scope,
        )
        result = await self.model_gateway.generate_text(
            task_definition=prompt_context.author_task_definition,
            task_input=writing_input,
            output_contract=output_contract,
            profile_id=profile_id,
            instance_id=instance_id,
            capability="text.completion",
            owner_kind="file_artifact",
            owner_id=str(task["task_id"]),
            idempotency_key=f"file-artifact:{task['task_id']}:body",
            work_purpose=AIWorkPurpose.FILE_GENERATION,
            logical_stage_key=f"file-artifact:{task['task_id']}:body",
            operation_timeout_seconds=timeout,
        )
        body = str(result.text or "").strip()
        if not body:
            await record_structured_rejection(
                model_gateway=self.model_gateway,
                completion=result,
                round_no=1,
                error="文件正文不能为空",
                terminal=True,
            )
            raise ValueError("文件正文模型返回了空正文")
        if body.startswith("```") and body.endswith("```"):
            await record_structured_rejection(
                model_gateway=self.model_gateway,
                completion=result,
                round_no=1,
                error="文件正文不得使用代码围栏",
                terminal=True,
            )
            raise ValueError("文件正文模型违反输出格式：不得使用代码围栏")
        body = self.identity.render(
            self.identity.decode_model(
                body,
                prompt_context.identity_catalog,
                scope=prompt_context.identity_scope,
            ),
            prompt_context.identity_context,
        ).strip()
        await record_structured_acceptance(
            model_gateway=self.model_gateway,
            completion=result,
            round_no=1,
            value={"body": body, "format": file_format},
        )
        return body

    async def _publish(
        self,
        task: dict[str, Any],
        control: Any,
        job: dict[str, Any],
        profile_id: str,
        instance_id: str,
        file_format: str,
        style: str,
        payload: dict[str, Any],
        body: str,
        images: dict[str, PDFImageAsset],
    ) -> dict[str, Any]:
        artifact = None
        try:
            await self._require_enabled_or_pause(profile_id, control)
            artifact = await self._generate_artifact(
                profile_id=profile_id,
                instance_id=instance_id,
                job_id=str(payload.get("job_id") or job["job_id"]),
                file_format=file_format,
                display_name=str(payload.get("display_name") or "生成文件"),
                content=body,
                document_style=style,
                image_assets=images,
            )
            await control.check_control()
            await self._require_enabled_or_pause(profile_id, control)
            published = await self.files.complete_file_generation_job(
                int(task["task_id"]),
                lease_token=int(task["lease_token"]),
                artifact=asdict(artifact),
            )
        except FileArtifactsDisabled:
            if artifact is not None and await self._not_committed(int(task["task_id"])):
                await asyncio.to_thread(self.service.release, artifact.storage_relpath)
            await control.pause(FILE_ARTIFACTS_DISABLED_REASON)
            raise AssertionError("unreachable after file task pause") from None
        except BaseException:
            if artifact is not None and await self._not_committed(int(task["task_id"])):
                await asyncio.to_thread(self.service.release, artifact.storage_relpath)
            raise
        return {
            "job_id": str(job["job_id"]),
            "todo_id": str(published["todo_id"]),
            "file_format": artifact.file_format,
            "byte_size": artifact.byte_size,
        }

    async def _generate_artifact(self, **values: Any) -> Any:
        """Render off-loop while retaining ownership of a cancelled thread."""

        generation = asyncio.create_task(
            asyncio.to_thread(self.service.generate, **values),
            name="soulcore-file-artifact-render",
        )
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                artifact = await asyncio.shield(generation)
                break
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
                if not generation.done():
                    continue
                if generation.cancelled():
                    raise cancellation from None
                try:
                    artifact = generation.result()
                except Exception:
                    raise cancellation from None
                break
        if cancellation is not None:
            try:
                await asyncio.to_thread(self.service.release, artifact.storage_relpath)
            except Exception as cleanup_error:
                cancellation.add_note(
                    "file artifact cleanup failed after cancelled render: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise cancellation
        return artifact

    async def _require_enabled_or_pause(self, profile_id: str, control: Any) -> None:
        if await self.files.get_profile_file_artifacts_enabled(profile_id):
            return
        await control.pause(FILE_ARTIFACTS_DISABLED_REASON)

    async def _not_committed(self, task_id: int) -> bool:
        try:
            job = await self.files.get_file_generation_job_for_task(task_id)
        except Exception:
            return False
        return job is not None and not (
            str(job.get("status") or "") == "SUCCEEDED"
            and job.get("todo_id")
            and job.get("file_asset_id")
        )

    @staticmethod
    def _body_output_contract(
        file_format: str,
        image_references: Sequence[str] = (),
    ) -> str:
        normalized = str(file_format or "").strip().upper()
        common = "只输出正文。不要出现写作过程、任务说明或任何元讨论。整份输出不用代码围栏包裹。"
        if normalized == "TXT":
            return (
                common + "纯文本格式。允许自然标题和必要的字符列表，"
                "不使用 Markdown 标题符号、表格分隔线、链接语法或图片语法。"
            )
        if normalized == "MD":
            return (
                common + "Markdown 格式。可用标题、段落、粗体、斜体、有序与无序列表、"
                "引用、分隔线、代码块、表格、行内代码和 http/https 链接。"
            )
        if normalized == "PDF":
            contract = (
                common + "Markdown 供 PDF 排版。标题使用一至三级；可用段落、粗体、斜体、"
                "有序与无序列表、引用、分隔线、代码块、表格、行内代码和 http/https 链接。"
            )
            available = tuple(
                dict.fromkeys(
                    reference
                    for value in image_references
                    if (reference := str(value or "").strip().upper())
                    in {"I1", "I2", "I3", "I4", "I5"}
                )
            )
            if not available:
                return contract + "本次没有配图，不输出图片语法、路径或 URL。"
            references = "、".join(available)
            return (
                contract + f"可用配图引用：{references}。"
                "使用标准 Markdown 图片语法 `![中文图注](引用名)` 插入，"
                "图注须与对应配图说明一致，每个引用至多使用一次。"
                "未使用的图会自动进入文末“相关配图”区，无需手动处理。"
                "不输出可用引用之外的图片路径或 URL。"
            )
        raise ValueError("unsupported file format")

    @staticmethod
    def _prompt(
        file_format: str,
        layout_preference: str,
        payload: dict[str, Any],
        image_lines: list[str],
    ) -> TrustedPromptMarkup:
        fact_policy = str(payload.get("fact_policy") or "").strip()
        fact_rule = (
            (
                "在用途和硬性要求允许的范围内，你可以创作人物、情节、例子、表达细节；"
                "但凡涉及现实事实、数字、承诺、来源、引语，仍须有材料支持。"
            )
            if fact_policy == _CREATIVE_POLICY
            else (
                "素材和硬性要求是你唯一的事实依据。"
                "缺信息时可以诚实限定范围、用显式占位符标出、或调整结构绕开，"
                "但不能编造事实、数字、承诺、来源或引语。"
            )
        )
        blocks: list[TrustedPromptMarkup] = [
            prompt_markup_record(
                "文档委托",
                {
                    "成品格式": file_format,
                    "用途与目标": str(payload.get("purpose") or "").strip(),
                    "目标读者": str(payload.get("audience") or "").strip(),
                    "必须覆盖": str(payload.get("requirements") or "").strip(),
                    "素材与事实来源": str(payload.get("source_materials") or "").strip(),
                    "语气与作者口吻": str(payload.get("voice") or "").strip(),
                    "事实边界": fact_rule,
                    "版式取向": layout_preference,
                },
            )
        ]
        if image_lines:
            blocks.append(
                prompt_markup_block(
                    "可用配图",
                    join_prompt_markup(
                        FileArtifactTaskExecutor._image_prompt_record(value)
                        for value in image_lines
                    ),
                )
            )
        return join_prompt_markup(blocks)

    @staticmethod
    def _image_prompt_record(value: str) -> TrustedPromptMarkup:
        reference, description = FileArtifactTaskExecutor._image_prompt_values(value)
        return prompt_markup_record(
            "配图",
            {
                "图片": reference,
                "说明": description,
            },
        )

    @staticmethod
    def _image_prompt_values(value: str) -> tuple[str, str]:
        raw = str(value or "").strip().removeprefix("-").strip()
        reference, separator, description = raw.partition("：")
        if not separator:
            reference, separator, description = raw.partition(":")
        return (
            reference.strip() if separator else "",
            description.strip() if separator else raw,
        )

    @staticmethod
    def _projection_description(projection: Any) -> str:
        if projection is None:
            return ""
        return str(projection.history_projection or projection.visible_facts or "").strip()

    @staticmethod
    def _recovered(job: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": str(job["job_id"]),
            "todo_id": str(job["todo_id"]),
            "file_asset_id": str(job.get("file_asset_id") or ""),
            "recovered_after_publish": True,
        }


__all__ = ["FileArtifactTaskExecutor", "file_artifact_operation_timeout_seconds"]
