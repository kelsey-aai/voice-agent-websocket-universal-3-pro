"""
Zero-framework voice agent — raw WebSocket to AssemblyAI Universal-3.5 Pro Streaming.

This tutorial shows exactly what's happening under the hood:
  Mic ──► sounddevice ──► AssemblyAI Universal-3.5 Pro WebSocket ──► GPT-4o ──► ElevenLabs ──► speakers

No LiveKit, no Pipecat, no Vapi — just raw WebSockets and Python.
Great for understanding the full pipeline or embedding into any custom app.

The one feature worth calling out: after every agent reply we push the spoken
text back to AssemblyAI as `agent_context` via an UpdateConfiguration message.
The model then transcribes the user's next turn *in the context of what the
agent just said*, which is where Universal-3.5 Pro's ~10% WER reduction on
voice-agent audio comes from.

Run: python agent.py
"""

import asyncio
import json
import os
import queue
import sys

import httpx
import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
ASSEMBLYAI_API_KEY = os.environ["ASSEMBLYAI_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

SAMPLE_RATE = 16000       # 16 kHz PCM
CHANNELS = 1
CHUNK_MS = 50             # ~50 ms audio chunks (recommended block size)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)

SPEECH_MODEL = "universal-3-5-pro"

# Universal-3.5 Pro Streaming WebSocket URL.
# Turn detection on Universal-3.5 Pro is punctuation-based and driven by the
# `mode` preset (min_latency | balanced | max_accuracy) plus the silence
# windows below — there is no end_of_turn_confidence_threshold on this model.
AAI_WS_URL = (
    "wss://streaming.assemblyai.com/v3/ws"
    f"?speech_model={SPEECH_MODEL}"
    "&encoding=pcm_s16le"           # 16-bit signed little-endian PCM
    f"&sample_rate={SAMPLE_RATE}"
    "&mode=balanced"                # balanced | min_latency | max_accuracy
    "&min_turn_silence=400"         # speculative end-of-turn check (ms)
    "&max_turn_silence=1600"        # hard cap before the turn is forced to end (ms)
    # "&voice_focus=near-field"     # (optional, +$0.10/hr) isolate the primary speaker
)

# Authenticate with your API key in the Authorization header (no "Bearer" prefix).
# This script runs server-side/locally, so the key is never exposed in a URL.
# For browser clients, generate a temporary token with GET /v3/token instead.
AAI_HEADERS = {"Authorization": ASSEMBLYAI_API_KEY}

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep every response under 2–3 sentences. "
    "Speak naturally — no markdown, no bullet points."
)


class VoiceAgent:
    def __init__(self):
        self.openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.is_speaking = False  # True while TTS is playing (for barge-in)

    # ── Microphone input ───────────────────────────────────────────────────

    def start_mic(self):
        """Capture mic audio into a queue using sounddevice."""

        def callback(indata: np.ndarray, frames: int, time, status):
            if status:
                print(f"Audio status: {status}", file=sys.stderr)
            # Convert float32 → int16 PCM
            pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            self.audio_queue.put(pcm)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=callback,
        )
        self.stream.start()
        print("🎙️  Microphone open — speak now")

    def stop_mic(self):
        self.stream.stop()
        self.stream.close()

    # ── AssemblyAI WebSocket ───────────────────────────────────────────────

    async def send_audio(self, ws):
        """Drain the mic queue and forward raw PCM bytes to AssemblyAI."""
        loop = asyncio.get_event_loop()
        while True:
            try:
                chunk = await loop.run_in_executor(
                    None, lambda: self.audio_queue.get(timeout=0.05)
                )
                await ws.send(chunk)
            except queue.Empty:
                await asyncio.sleep(0.01)

    async def receive_transcripts(self, ws):
        """Receive turn transcripts from AssemblyAI and respond."""
        async for raw in ws:
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "Begin":
                applied = msg.get("configuration", {}) or {}
                model = applied.get("model")
                print(f"✅ AssemblyAI session: {msg.get('id', 'unknown')}")
                # Bad/misspelled query params are ignored, not rejected —
                # so confirm the model that was actually applied.
                if model != SPEECH_MODEL:
                    print(f"⚠️  Expected {SPEECH_MODEL}, got {model}")
                print()

            elif msg_type == "Turn":
                transcript = msg.get("transcript", "").strip()
                end_of_turn = msg.get("end_of_turn", False)

                if transcript:
                    print(f"\r👤 {transcript}", end="", flush=True)

                if end_of_turn and transcript:
                    print()  # newline after the rolling partial
                    await self.handle_turn(ws, transcript)

            elif msg_type == "Termination":
                print("\n🔇 Session terminated")
                break

    async def handle_turn(self, ws, user_text: str):
        """Generate an LLM response, speak it, and feed it back as agent_context."""
        self.conversation.append({"role": "user", "content": user_text})

        # ── LLM ──────────────────────────────────────────────────────────
        response = await self.openai.chat.completions.create(
            model="gpt-4o",
            messages=self.conversation,
            max_tokens=150,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        self.conversation.append({"role": "assistant", "content": reply})
        print(f"🤖 {reply}\n")

        # ── Feed the reply back as context for the user's NEXT turn ────────
        # Universal-3.5 Pro uses this to hear short/ambiguous replies and
        # spelled-out entities (emails, IDs) in the context of what the
        # agent just asked. Each UpdateConfiguration replaces the prior value.
        await ws.send(json.dumps({"type": "UpdateConfiguration", "agent_context": reply}))

        # ── TTS ───────────────────────────────────────────────────────────
        self.is_speaking = True
        await self.speak(reply)
        self.is_speaking = False

    async def speak(self, text: str):
        """Synthesise speech with ElevenLabs and play it through the speakers."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_turbo_v2",
                    "output_format": "pcm_16000",
                },
                timeout=15,
            )
            if resp.status_code != 200:
                print(f"TTS error {resp.status_code}: {resp.text}")
                return

            audio_data = np.frombuffer(resp.content, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(audio_data, samplerate=SAMPLE_RATE, blocking=True)

    # ── Main run loop ────────────────────────────────────────────────────────

    async def run(self):
        print("🚀 Voice Agent — AssemblyAI Universal-3.5 Pro Streaming")
        print("   Press Ctrl+C to quit\n")

        self.start_mic()

        try:
            async with websockets.connect(AAI_WS_URL, additional_headers=AAI_HEADERS) as ws:
                try:
                    await asyncio.gather(
                        self.send_audio(ws),
                        self.receive_transcripts(ws),
                    )
                finally:
                    # Close the AssemblyAI session cleanly.
                    await ws.send(json.dumps({"type": "Terminate"}))
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
        finally:
            self.stop_mic()


async def main():
    agent = VoiceAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
