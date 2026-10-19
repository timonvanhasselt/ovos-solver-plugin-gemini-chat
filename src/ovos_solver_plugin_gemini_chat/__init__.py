from ovos_gemini_plugin.chat import GeminiChatEngine

# ready-made persona exposed via the "opm.plugin.persona" entry point
GEMINI_DEMO = {
    "name": "Gemini Chat",
    "solvers": [  # key name kept for persona compatibility; accepts chat engines
        "ovos-chat-gemini-plugin"
    ],
    "ovos-chat-gemini-plugin": {
        "api_key": "your-gemini-api-key",
        "model": "gemini-2.5-flash",
        "enable_grounding": True,
    }
}

__all__ = ["GeminiChatEngine", "GEMINI_DEMO"]
