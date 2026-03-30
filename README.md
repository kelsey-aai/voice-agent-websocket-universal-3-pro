# Raw WebSocket voice agent with AssemblyAI Universal-3 Pro Streaming

The simplest possible voice agent — no frameworks, no abstraction layers. Just raw WebSockets, a microphone, and the **AssemblyAI Universal-3 Pro Streaming model** (`u3-rt-pro`).

This tutorial shows exactly what LiveKit Agents, Pipecat, and Vapi are doing underneath. If you want full control over every byte, or you're embedding a voice agent into a custom application, start here.

## The pipeline

```
Microphone
    │ float32 audio (sounddevice)
    ▼ convert → int16 PCM
    │
AssemblyAI WebSocket (wss://streaming.assemblyai.com/v3/ws)
    │ ?speech_model=u3-rt-pro&encoding=pcm_s16le&sample_rate=16000
    │
    │ Turn message (end_of_turn=true) — neural turn detection
    ▼
OpenAI GPT-4o
    │ text response
    ▼
ElevenLabs TTS (pcm_16000 format)
    │ PCM audio bytes
    ▼
Speakers (sounddevice)
```

## Prerequisites

- Python 3.11+
- A microphone and speakers
- [AssemblyAI API key](https://app.assemblyai.com)
- [OpenAI API key](https://platform.openai.com/api-keys)
- [ElevenLabs API key](https://elevenlabs.io)

On macOS, install PortAudio for sounddevice:

```bash
brew install portaudio
```

## Quick start

```bash
git clone https://github.com/kelseyefoster/voice-agent-websocket-universal-3-pro
cd voice-agent-websocket-universal-3-pro

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys

python agent.py
```

Speak — partial transcripts appear in real time. When Universal-3 Pro detects an end-of-turn, the agent responds.

## How it works

### 1. WebSocket connection

```python
AAI_WS_URL = (
    "wss://streaming.assemblyai.com/v3/ws"
    "?speech_model=u3-rt-pro"
    "&encoding=pcm_s16le"           # 16-bit signed little-endian PCM
    "&sample_rate=16000"            # 16kHz
    "&end_of_turn_confidence_threshold=0.4"
    f"&token={ASSEMBLYAI_API_KEY}"
)
```

### 2. Message types

AssemblyAI v3 sends three event types:

```json
// Session started
{ "type": "Begin", "id": "session_abc123" }

// Rolling transcript — update your UI with this
{ "type": "Turn", "transcript": "how do I", "end_of_turn": false }

// End-of-turn detected — respond now
{ "type": "Turn", "transcript": "how do I get started?", "end_of_turn": true,
  "words": [{"text": "how", "start": 0, "end": 200, "confidence": 0.99}, ...] }

// Session closed
{ "type": "Termination" }
```

### 3. Sending audio

```python
# Raw PCM bytes — no wrapper, no base64
await ws.send(pcm_bytes)

# Terminate the session cleanly
await ws.send(json.dumps({"type": "Terminate"}))
```

## Tuning turn detection

| Setting | Effect |
|---------|--------|
| Lower `end_of_turn_confidence_threshold` (e.g. `0.3`) | Faster response, more false triggers |
| Higher `end_of_turn_confidence_threshold` (e.g. `0.6`) | More patient, better for noisy environments |
| Lower `min_turn_silence` (e.g. `200ms`) | Snappier for fast-paced conversation |
| Higher `max_turn_silence` (e.g. `2000ms`) | Better for deliberate speech or elderly users |

## Swapping components

```python
# Anthropic Claude instead of GPT-4o
from anthropic import AsyncAnthropic
client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
response = await client.messages.create(model="claude-opus-4-6", max_tokens=150, ...)

# Cartesia instead of ElevenLabs (lower TTS latency)
import cartesia
```

## Related tutorials

- [Tutorial 01: LiveKit + Universal-3 Pro Streaming](../01-livekit-universal-3-pro) — production-ready framework built on top of exactly this WebSocket pattern
- [Tutorial 02: Pipecat + Universal-3 Pro Streaming](../02-pipecat-universal-3-pro) — modular pipeline that adds VAD, turn handling, and transport management
- [Tutorial 08: Node.js voice agent](../08-nodejs-assemblyai) — the same raw WebSocket approach in JavaScript

## Resources

- [AssemblyAI Universal Streaming docs](https://www.assemblyai.com/docs/speech-to-text/universal-streaming)
- [Universal-3 Pro Streaming API reference](https://www.assemblyai.com/docs/api-reference/streaming)
- [AssemblyAI Python SDK](https://github.com/AssemblyAI/assemblyai-python-sdk)

---

<div class="blog-cta_component">
  <div class="blog-cta_title">Build your first voice agent from scratch</div>
  <div class="blog-cta_rt w-richtext">
    <p>Sign up for a free AssemblyAI account and connect to the Universal-3 Pro Streaming WebSocket in minutes. No framework required.</p>
  </div>
  <a href="https://www.assemblyai.com/dashboard/signup" class="button w-button">Start building</a>
</div>

<div class="blog-cta_component">
  <div class="blog-cta_title">Experiment with real-time turn detection</div>
  <div class="blog-cta_rt w-richtext">
    <p>Try streaming transcription in our Playground and observe how punctuation and silence handling shape turn boundaries in real time. Compare behaviors across Universal-3 Pro Streaming and Universal-streaming models.</p>
  </div>
  <a href="https://www.assemblyai.com/playground" class="button w-button">Open playground</a>
</div>
