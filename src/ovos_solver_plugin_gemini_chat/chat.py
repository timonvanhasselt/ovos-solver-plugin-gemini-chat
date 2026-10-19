from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from google.genai import Client, types
from ovos_plugin_manager.templates.agents import (
    AgentMessage,
    ChatEngine,
    MessageRole,
)
from ovos_utils.log import LOG
from sentence_stream import SentenceBoundaryDetector

# Characters that TTS engines tend to read out loud or choke on (markdown, emoji, ...)
NON_TEXT_CHARS = re.compile(r"[^\w\s\.\,\?\!\:\;\-\&\'\"]")


def clean_for_tts(text: str) -> str:
    """Strip markdown/emoji/symbols so the text can be spoken."""
    return NON_TEXT_CHARS.sub("", text)


class GeminiChatEngine(ChatEngine):
    """
    ChatEngine for the Google Gemini API (``google-genai`` SDK).

    The persona owns the conversation history and passes it in on every call
    as a list of ``AgentMessage``. This engine is therefore stateless: it
    converts the messages to Gemini ``Content`` objects and sends them with
    ``generate_content`` (no server-side ``chats`` session).

    Configuration (``config``):
        api_key (str): Gemini API key (required).
        model (str): model name, default ``gemini-2.5-flash``.
        system_prompt (str): system instruction sent with every request.
        allow_system_prompts (bool): keep system messages from the persona /
            memory plugins and merge them into the system instruction.
            Default False (they are dropped, like ovos-chat-openai-plugin).
        enable_reasoning (bool): thinking on (dynamic budget) or off.
        enable_grounding (bool): enable Google Search grounding.
        temperature (float), max_output_tokens (int): optional.
    """

    supports_tools = False  # Gemini function calling is not wired up here
    priority = 50  # ChatEngine declares no default; personas sort on it

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        if not self.api_key:
            LOG.error("'api_key' not set in config")
            raise ValueError("api key must be set")

        self.model = self.config.get("model", "gemini-2.5-flash")
        self.reasoning = self.config.get("enable_reasoning", False)
        self.grounding = self.config.get("enable_grounding", False)
        self.allow_system = self.config.get("allow_system_prompts") or False
        self.system_prompt = (
            self.config.get("system_prompt")
            or self.config.get("initial_prompt")
            or "You are a helpful assistant."
        )
        self._warned_dropped_system = False

        self.client = Client(api_key=self.api_key)
        self._check_model()
        LOG.info(f"Using Gemini model: {self.model} (grounding: {self.grounding})")

    # ------------------------------------------------------------------ setup
    def _check_model(self) -> None:
        """Log available models; never change the configured model."""
        try:
            available = [
                m.name.split("/")[-1]
                for m in self.client.models.list()
                if m.name and "gemini" in m.name
            ]
            LOG.debug(f"Available Gemini models: {available}")
            if available and self.model not in available:
                LOG.warning(f"model '{self.model}' not in the list of available models")
        except Exception as e:  # listing is informational only
            LOG.warning(f"unable to list Gemini models: {e}")

    def _build_config(self, system_instruction: str) -> types.GenerateContentConfig:
        tools = [types.Tool(google_search=types.GoogleSearch())] if self.grounding else None
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1 if self.reasoning else 0
            ),
            system_instruction=system_instruction,
            tools=tools,
            temperature=self.config.get("temperature"),
            max_output_tokens=self.config.get("max_output_tokens"),
        )

    # --------------------------------------------------------------- messages
    def _prepare(self, messages: List[AgentMessage]) -> Tuple[str, List[types.Content]]:
        """AgentMessage list -> (system instruction, Gemini contents)."""
        system_parts = [self.system_prompt] if self.system_prompt else []
        contents: List[types.Content] = []
        dropped = 0

        for m in messages:
            text = (m.content or "").strip()
            if m.role in (MessageRole.SYSTEM, MessageRole.DEVELOPER):
                if self.allow_system:
                    if text:
                        system_parts.append(text)
                else:
                    dropped += 1
                continue
            if m.role == MessageRole.USER:
                role = "user"
            elif m.role == MessageRole.ASSISTANT:
                role = "model"
            else:  # TOOL etc.: not supported by this engine
                continue
            if not text:
                continue
            # Gemini expects alternating turns: merge consecutive same-role messages
            if contents and contents[-1].role == role:
                contents[-1].parts.append(types.Part(text=text))
            else:
                contents.append(types.Content(role=role, parts=[types.Part(text=text)]))

        # the conversation must start with a user turn
        while contents and contents[0].role != "user":
            contents.pop(0)

        if dropped and not self._warned_dropped_system:
            self._warned_dropped_system = True
            LOG.warning(
                f"dropped {dropped} system message(s): 'allow_system_prompts' is off. "
                f"Context injected by a memory plugin as a system message is discarded too."
            )
        return "\n".join(system_parts), contents

    # ---------------------------------------------------------------- logging
    @staticmethod
    def _log_grounding(response: Any) -> None:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return
        meta = getattr(candidates[0], "grounding_metadata", None)
        if not meta:
            return
        if getattr(meta, "web_search_queries", None):
            LOG.debug(f"Gemini search queries: {meta.web_search_queries}")
        for i, chunk in enumerate(getattr(meta, "grounding_chunks", None) or [], 1):
            if getattr(chunk, "web", None):
                LOG.debug(f"Source {i}: {chunk.web.title} -> {chunk.web.uri}")

    # -------------------------------------------------------- ChatEngine API
    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None,
                      tools: Any = None) -> AgentMessage:
        system, contents = self._prepare(messages)
        if not contents:
            return AgentMessage(role=MessageRole.ASSISTANT, content="")
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._build_config(system),
        )
        self._log_grounding(response)
        return AgentMessage(role=MessageRole.ASSISTANT,
                            content=(response.text or "").strip())

    def stream_tokens(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> Iterable[str]:
        system, contents = self._prepare(messages)
        if not contents:
            return
        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=self._build_config(system),
        )
        for chunk in stream:
            self._log_grounding(chunk)
            if chunk.text:
                yield chunk.text

    def stream_sentences(self, messages: List[AgentMessage],
                         session_id: str = "default",
                         lang: Optional[str] = None,
                         units: Optional[str] = None) -> Iterable[str]:
        detector = SentenceBoundaryDetector()
        for tok in self.stream_tokens(messages, session_id, lang, units):
            for sentence in detector.add_chunk(tok):
                sentence = clean_for_tts(sentence).strip()
                if sentence:
                    yield sentence
        final = clean_for_tts(detector.finish() or "").strip()
        if final:
            yield final
