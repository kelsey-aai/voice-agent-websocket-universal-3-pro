"""
Zero-framework voice agent — raw WebSocket to AssemblyAI Universal-3 Pro Streaming.

This tutorial shows exactly what's happening under the hood:
  Mic ──► sounddevice ──► AssemblyAI U3 Pro WebSocket ──► GPT-4o ──► ElevenLabs ──► speakers

No LiveKit, no Pipecat, no Vapi — just raw WebSockets and Python.
Great for understanding the full pipeline or embedding into any custom app.

Run: python agent.py
"""

import asyncio
import base64
import json
import os
import queue
import sys
import threading

import httpx
import sounddevice as sd
import numpy as np
import websockets
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
ASSEMBLYAI_API_KEY = os.environ["ASSEMBLYAI_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

SAMPLE_RATE = 16000       # 16kHz PCM — optimal for U3 Pro
CHANNELS = 1
CHUNK_MS = 100            # Send 100ms audio chunks
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)

# Universal-3 Pro Streaming WebSocket URL
AAI_WS_URL = (
    "wss://streaming.assemblyai.com/v3/ws"
    "?speech_model=u3-rt-pro"
    "&encoding=pcm_s16le"
    f"&sample_rate={SAMPLE_RATE}"
    "&end_of_turn_confidence_threshold=0.4"
    "&min_turn_silence=300"
    "&max_turn_silence=1200"
    f"&token={ASSEMBLYAI_API_KEY}"
)

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
        """Capture mic audio into queue using sounddevice."""

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
        """Drain the mic queue and forward PCM bytes to AssemblyAI."""
        loop = asyncio.get_event_loop()
        while True:
            # Non-blocking get with asyncio-compatible sleep
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
            msg_type = msg.get("message_type") or msg.get("type", "")

            if msg_type == "Begin":
                session_id = msg.get("id", "unknown")
                print(f"✅ AssemblyAI session: {session_id}\n")

            elif msg_type == "Turn":
                transcript = msg.get("transcript", "").strip()
                end_of_turn = msg.get("end_of_turn", False)

                if transcript:
                    # Show partial transcripts in grey
                    print(f"\r👤 {transcript}", end="", flush=True)

                if end_of_turn and transcript:
                    print()  # newline after partial
                    await self.handle_turn(transcript)

            elif msg_type == "Termination":
                print("\n🔇 Session terminated")
                break

    async def handle_turn(self, user_text: str):
        """Generate LLM response and speak it."""
        self.conversation.append({"role": "user", "content": user_text})

        # ── LLM ─────────────────────────────────────────────────────────
        response = await self.openai.chat.completions.create(
            model="gpt-4o",
            messages=self.conversation,
            max_tokens=150,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        self.conversation.append({"role": "assistant", "content": reply})
        print(f"🤖 {reply}\n")

        # ── TTS ──────────────────────────────────────────────────────────
        self.is_speaking = True
        await self.speak(reply)
        self.is_speaking = False

    async def speak(self, text: str):
        """Synthesise speech with ElevenLabs and play through speakers."""
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

    # ── Main run loop ──────────────────────────────────────────────────────

    async def run(self):
        print("🚀 Voice Agent — AssemblyAI Universal-3 Pro Streaming")
        print("   Press Ctrl+C to quit\n")

        self.start_mic()

        try:
            async with websockets.connect(AAI_WS_URL) as ws:
                await asyncio.gather(
                    self.send_audio(ws),
                    self.receive_transcripts(ws),
                )
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
        finally:
            self.stop_mic()


async def main():
    agent = VoiceAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
