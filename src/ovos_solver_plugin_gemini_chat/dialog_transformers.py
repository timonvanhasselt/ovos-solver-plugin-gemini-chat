from __future__ import annotations

from typing import Any, Optional

from ovos_plugin_manager.templates.transformers import DialogTransformer
from ovos_utils.log import LOG

from ovos_gemini_plugin.chat import GeminiChatEngine


class GeminiDialogTransformer(DialogTransformer):
    """Rewrite dialog with Gemini just before TTS.

    Config (``dialog_transformers`` -> ``ovos-dialog-transformer-plugin-gemini-chat``):
        api_key / key (str): Gemini API key (required).
        model (str): default ``gemini-2.5-flash-lite``.
        rewrite_prompt (str): what to do with the text; can be overridden per
            dialog through ``context["prompt"]``.
        system_prompt (str): optional override of the system instruction.
    """

    def __init__(
        self,
        name: str = "ovos-dialog-transformer-plugin-gemini-chat",
        priority: int = 10,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, priority, config)
        self.engine: Optional[GeminiChatEngine] = None
        api_key = self.config.get("api_key") or self.config.get("key")
        if not api_key:
            LOG.error(f"{name}: no 'api_key' in config, dialog will not be rewritten")
            return
        self.engine = GeminiChatEngine(
            {
                "api_key": api_key,
                "model": self.config.get("model", "gemini-2.5-flash-lite"),
                "system_prompt": self.config.get("system_prompt") or (
                    "Your task is to rewrite text as if it was spoken by a "
                    "different character. Reply with the rewritten text only."
                ),
                # a rewrite must never search the web or "think"
                "enable_grounding": False,
                "enable_reasoning": False,
                "temperature": self.config.get("temperature"),
                "max_output_tokens": self.config.get("max_output_tokens"),
            }
        )

    def transform(
        self,
        dialog: str,
        context: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """
        Optionally transform passed dialog and/or return additional context
        :param dialog: str utterance to mutate before TTS
        :returns: str mutated dialog
        """
        context = context or {}
        prompt = context.get("prompt") or self.config.get("rewrite_prompt")
        if self.engine is None or not prompt:
            return dialog, context
        try:
            rewritten = self.engine.get_response(
                f"{prompt}: {dialog}", lang=context.get("lang")
            )
        except Exception as e:
            # never lose the original dialog because the API failed
            LOG.error(f"Gemini dialog rewrite failed: {e}")
            return dialog, context
        return rewritten or dialog, context
