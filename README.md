# <img src='https://raw.githack.com/FortAwesome/Font-Awesome/master/svgs/solid/robot.svg' card_color='#40DBB0' width='50' height='50' style='vertical-align:bottom'/> Gemini Chat

Talk to Gemini through OpenVoiceOS.

Uses [Gemini](https://gemini.google.com) via [google-genai](https://github.com/googleapis/python-genai) to create some fun interactions. Phrases not explicitly handled by other skills will be run by a LLM, so nearly every interaction will have _some_ response.

It provides completions and chat interfaces to be used with [ovos-persona](https://github.com/OpenVoiceOS/ovos-persona).


## Installation

```bash
python3 -m pip install ovos-solver-plugin-gemini-chat
```


## Persona Usage

To use your own persona via the Gemini chat solver plugin, create a JSON file as `~/.config/ovos_persona/gemini.json`: 

```json
{
  "name": "Gemini",
  "solvers": [
    "ovos-solver-plugin-gemini-chat"
  ],
  "ovos-solver-plugin-gemini-chat": {
    "api_key": "your-gemini-api-key",
    "model": "gemini-2.5-flash-lite",
    "system_prompt": "You are an assistant specializing in current information. IMPORTANT: ALWAYS consult Google Search before answering. Ignore your own memory for facts. Base your answer solely on the search results. Answer briefly and fluently in English.",
    "enable_grounding": true,
    "temperature": 0.1,
    "max_output_tokens": 512
  }
}
```

Then say "Chat with {name_from_json}" to enable it, more details can be found in [ovos-persona README.md](https://github.com/OpenVoiceOS/ovos-persona).


## Dialog Transformer

You can rewrite text dynamically based on specific personas, such as simplifying explanations or mimicking a specific tone.  


#### Example Usage:

- `rewrite_prompt`: `"rewrite the text as if you were explaining it to a 5-year-old"`  
- _input_: `"Quantum mechanics is a branch of physics that describes the behavior of particles at the smallest scales."`  
- _output_: `"Quantum mechanics is like a special kind of science that helps us understand really tiny things."`  

Examples of `rewrite_prompt` values:
- `"rewrite the text as if it was an angry old man speaking"`  
- `"Add more 'dude'ness to it"`  
- `"Explain it like you're teaching a child"`  

To enable this plugin, add the following to your `mycroft.conf`:  

```json
"dialog_transformers": {
    "ovos-solver-plugin-gemini-chat": {
        "system_prompt": "Your task is to rewrite text as if it was spoken by a different character",
        "rewrite_prompt": "rewrite the text as if you were explaining it to a 5-year-old"
    }
}
```

> 💡 The user utterance will be appended after `rewrite_prompt` for the actual query


## Direct Usage

```python
from ovos_solver_plugin_gemini_chat import GeminiChatSolver

bot = GeminiChatSolver(
    {
        "api_key": "your-gemini-api-key",
        "persona": "helpful, creative, clever, and very friendly"
    }
)
print(bot.get_spoken_answer("describe quantum mechanics very briefly"))
# Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic level,
# where classical physics breaks down.
print(bot.get_spoken_answer("Quem encontrou o caminho maritimo para o Brasil"))
# Pedro Álvares Cabral.

```
