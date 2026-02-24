import json
import os
from datetime import datetime, timezone
from pathlib import Path

from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.core import logger
from astrbot.core.utils.astrbot_path import get_astrbot_data_path


class Main(star.Star):
    """审计日志插件：记录 LLM 实际看到的完整上下文。"""

    def __init__(self, context: star.Context) -> None:
        self.context = context
        base = Path(get_astrbot_data_path()) / "audit_logs"
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("创建审计日志目录失败: %s", exc)
        self.base_dir = base

    def _build_context_snapshot(self, req: ProviderRequest) -> list[dict]:
        """尽量还原 LLM 实际收到的 messages 列表。"""
        messages: list[dict] = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})

        ctx = req.contexts or []
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except Exception:
                ctx = []

        if isinstance(ctx, list):
            for m in ctx:
                if isinstance(m, dict):
                    messages.append(m)

        return messages

    async def _append_current_user_message(
        self,
        req: ProviderRequest,
        messages: list[dict],
    ) -> None:
        """把当前用户消息追加到 context_snapshot 末尾。"""
        try:
            user_msg = await req.assemble_context()
        except Exception as exc:  # noqa: BLE001
            logger.warning("组装用户消息失败，将使用简化版本: %s", exc)
            if req.prompt:
                user_msg = {"role": "user", "content": req.prompt}
            else:
                return
        messages.append(user_msg)

    def _get_log_path(self) -> str:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.base_dir, f"{day}.jsonl")

    def _safe_dumps(self, payload: dict) -> str:
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("审计日志序列化失败: %s", exc)
            return "{}"

    @filter.on_llm_response()
    async def on_llm_response(
        self,
        event: AstrMessageEvent,
        response: LLMResponse,
    ) -> None:
        """在每轮 LLM 响应后写入一条审计记录。"""
        req: ProviderRequest | None = event.get_extra("provider_request")
        if not req:
            return

        context_snapshot = self._build_context_snapshot(req)
        await self._append_current_user_message(req, context_snapshot)

        mem0_injected = event.get_extra("mem0_injected") or []
        window = event.get_extra("short_term_window")
        pending_count = getattr(window, "pending_count", 0) if window else 0

        # 通过 Context 获取当前 provider 以拿到模型名（可能为空）
        model = None
        try:
            provider = self.context.get_using_provider(event.unified_msg_origin)
            if provider:
                model = provider.get_model()
        except Exception:
            model = None

        token_usage = response.usage.total if response.usage else None

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": event.unified_msg_origin,
            "user_id": event.get_sender_id(),
            "user_message": req.prompt,
            "assistant_reply": response.completion_text,
            "mem0_injected": mem0_injected,
            "context_snapshot": context_snapshot,
            "pending_count": pending_count,
            "model": model,
            "token_usage": token_usage,
        }

        line = self._safe_dumps(record)
        path = self._get_log_path()
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:  # noqa: BLE001
            logger.error("写入审计日志失败(%s): %s", path, exc)

