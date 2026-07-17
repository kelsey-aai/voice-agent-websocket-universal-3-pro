# Raw WebSocket voice agent with AssemblyAI Universal-3.5 Pro Streaming

The simplest possible voice agent — no frameworks, no abstraction layers. Just raw WebSockets, a microphone, and the **AssemblyAI Universal-3.5 Pro Streaming model** (`universal-3-5-pro`).

This tutorial shows exactly what LiveKit Agents, Pipecat, and Vapi are doing underneath. If you want full control over every byte, or you're embedding a voice agent into a custom application, start here.

## The pipeline

```
Microphone
    │ float32 audio (sounddevice)
    ▼ convert → int16 PCM
    │
AssemblyAI WebSocket (wss://streaming.assemblyai.com/v3/ws)
    │ ?speech_model=universal-3-5-pro&encoding=pcm_s16le&sample_rate=16000&mode=balanced
    │
    │ Turn message (end_of_turn=true) — punctuation-based turn detection
    ▼
OpenAI GPT-4o
    │ text response
    │──► UpdateConfiguration { agent_context } back to AssemblyAI
    ▼
ElevenLabs TTS (pcm_16000 format)
    │ PCM audio bytes
    ▼
Speakers (sounddevice)
```

## Prerequisites

- Python 3.11+
- A microphone and speakers
- [AssemblyAI API key](https://www.assemblyai.com/dashboard/signup)
- [OpenAI API key](https://platform.openai.com/api-keys)
- [ElevenLabs API key](https://elevenlabs.io)

On macOS, install PortAudio for sounddevice:

```bash
brew install portaudio
```

## Quick start

```bash
git clone https://github.com/kelsey-aai/voice-agent-websocket-universal-3-5-pro
cd voice-agent-websocket-universal-3-5-pro

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys

python agent.py
```

Speak — partial transcripts appear in real time. When Universal-3.5 Pro detects an end-of-turn, the agent responds, then feeds its reply back to the model as context for your next turn.

## How it works

### 1. WebSocket connection

Connect to the v3 endpoint with your speech model and audio format. Authenticate with your API key in the `Authorization` header (no `Bearer` prefix) — the key never appears in a URL.

```python
SPEECH_MODEL = "universal-3-5-pro"

AAI_WS_URL = (
    "wss://streaming.assemblyai.com/v3/ws"
    f"?speech_model={SPEECH_MODEL}"
    "&encoding=pcm_s16le"           # 16-bit signed little-endian PCM
    "&sample_rate=16000"            # 16 kHz
    "&mode=balanced"                # balanced | min_latency | max_accuracy
    "&min_turn_silence=400"
    "&max_turn_silence=1600"
)

async with websockets.connect(
    AAI_WS_URL, additional_headers={"Authorization": ASSEMBLYAI_API_KEY}
) as ws:
    ...
```

> Unrecognized or misspelled query parameters are **ignored, not rejected**. The `Begin` message echoes the applied config — check that `configuration.model` matches `universal-3-5-pro`.

### 2. Message types

AssemblyAI v3 sends JSON events keyed on `type`:

```json
// Session started — echoes the applied configuration
{ "type": "Begin", "id": "3207b601-...", "expires_at": 1772570132,
  "configuration": { "model": "universal-3-5-pro", "mode": "balanced" } }

// Speech detected (always followed by Turn messages)
{ "type": "SpeechStarted", "timestamp": 1200, "confidence": 0.98 }

// Rolling transcript — update your UI with this
{ "type": "Turn", "transcript": "how do I", "end_of_turn": false }

// End-of-turn detected — respond now
{ "type": "Turn", "transcript": "how do I get started?", "end_of_turn": true,
  "turn_order": 0, "words": [{"text": "how", "start": 0, "end": 200, "confidence": 0.99}] }

// Session closed
{ "type": "Termination", "audio_duration_seconds": 12.4, "session_duration_seconds": 30.1 }
```

### 3. Sending audio

```python
# Raw PCM bytes — no wrapper, no base64
await ws.send(pcm_bytes)

# Terminate the session cleanly
await ws.send(json.dumps({"type": "Terminate"}))
```

### 4. Context carryover with `agent_context`

This is the Universal-3.5 Pro feature that matters most for voice agents. After each agent reply, push the spoken text back with an `UpdateConfiguration` message. The model then transcribes the user's next turn in the context of what the agent just said — which cuts word error rate ~10% on voice-agent audio and dramatically improves short replies and spelled-out entities (emails, IDs, confirmation numbers).

```python
# After generating (and before speaking) the agent's reply:
await ws.send(json.dumps({
    "type": "UpdateConfiguration",
    "agent_context": reply,   # e.g. "Sure — what date would you like to book?"
}))
```

Each `UpdateConfiguration` replaces the previously set `agent_context`. A short rolling conversation memory is on by default (no config needed).

## Tuning turn detection

Turn detection on Universal-3.5 Pro is **punctuation-based** and driven by the `mode` preset plus two silence windows. (Note: `end_of_turn_confidence_threshold` applies to the older Universal-Streaming models, **not** Universal-3.5 Pro.)

| Setting | Effect |
|---------|--------|
| `mode=min_latency` | Snappiest responses; best for fast back-and-forth |
| `mode=max_accuracy` | Most patient/accurate; built for drive-thru & noisy rooms |
| Lower `min_turn_silence` (e.g. `200`) | Faster end-of-turn checks |
| Higher `max_turn_silence` (e.g. `2000`) | Better for deliberate speakers |
| Raise `vad_threshold` (e.g. `0.4`) | Fewer false triggers in noisy environments |
| `voice_focus=near-field` / `far-field` | Isolate the primary speaker (+$0.10/hr) |

You can update `min_turn_silence`, `max_turn_silence`, `prompt`, `keyterms_prompt`, `agent_context`, and `language_codes` mid-stream via `UpdateConfiguration`. Send `{"type": "ForceEndpoint"}` to end the current turn immediately.

## Swapping components

```python
# Anthropic Claude instead of GPT-4o
from anthropic import AsyncAnthropic
client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
response = await client.messages.create(model="claude-opus-4-8", max_tokens=150, ...)

# Cartesia instead of ElevenLabs (lower TTS latency)
import cartesia
```

## Prefer not to wire it up yourself?

This repo is the DIY path — you own the STT + LLM + TTS glue. If you'd rather ship faster, the [AssemblyAI Voice Agent API](https://www.assemblyai.com/blog/introducing-our-voice-agent-api) bundles all three behind a single WebSocket at a flat $4.50/hr, built on this same Universal-3.5 Pro model.

## Resources

- [AssemblyAI Universal Streaming docs](https://www.assemblyai.com/docs/speech-to-text/universal-streaming)
- [Context Carryover (agent_context) guide](https://www.assemblyai.com/docs/streaming/universal-3-pro/context-carryover)
- [Voice Focus](https://www.assemblyai.com/docs/streaming/voice-focus)
- [Streaming WebSocket API reference](https://www.assemblyai.com/docs/api-reference/streaming)
- [AssemblyAI Python SDK](https://github.com/AssemblyAI/assemblyai-python-sdk)
