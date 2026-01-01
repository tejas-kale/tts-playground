"""Runpod handler for ChatterboxTurboTTS.

This handler provides serverless TTS inference using ChatterboxTurboTTS.
"""

from __future__ import annotations

import base64
import io
import os
import re
from typing import Any

import runpod
import torch
import torchaudio as ta


# Global model (loaded once on cold start)
chatterbox_model = None


def load_chatterbox():
    """Load the Chatterbox TTS model."""
    global chatterbox_model

    if chatterbox_model is None:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        from huggingface_hub import login

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN required for Chatterbox")

        print("Loading Chatterbox Turbo TTS model...")
        login(token=hf_token)
        chatterbox_model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        print("Chatterbox model loaded successfully!")

    return chatterbox_model


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    tokens = tokenizer.encode(text)
    return len(tokens)


def split_text_into_chunks(text: str, tokenizer, max_tokens: int = 100) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence, tokenizer)

        if sentence_tokens > max_tokens:
            parts = re.split(r'(?<=,)\s+', sentence)
            for part in parts:
                part_tokens = count_tokens(part, tokenizer)
                if current_tokens + part_tokens > max_tokens:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [part]
                        current_tokens = part_tokens
                    else:
                        chunks.append(part)
                else:
                    current_chunk.append(part)
                    current_tokens += part_tokens
        else:
            if current_tokens + sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def synthesize_chatterbox(text: str) -> bytes:
    """Synthesize speech using ChatterboxTurboTTS."""
    model = load_chatterbox()

    # Get tokenizer
    if hasattr(model, "text_encoder"):
        tokenizer = model.text_encoder.tokenizer
    elif hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
    elif hasattr(model, "text_embedder") and hasattr(model.text_embedder, "tokenizer"):
        tokenizer = model.text_embedder.tokenizer
    else:
        raise AttributeError("Could not find tokenizer in ChatterboxTurboTTS model")

    # Check token count
    total_tokens = count_tokens(text, tokenizer)

    if total_tokens <= 100:
        print(f"Generating audio directly ({total_tokens} tokens)...")
        wav = model.generate(text)
    else:
        print(f"Text has {total_tokens} tokens, splitting into chunks...")
        chunks = split_text_into_chunks(text, tokenizer, max_tokens=100)
        print(f"Split into {len(chunks)} chunks")

        audio_chunks = []
        for i, chunk in enumerate(chunks, 1):
            chunk_tokens = count_tokens(chunk, tokenizer)
            print(f"  Chunk {i}/{len(chunks)}: {chunk_tokens} tokens")
            wav = model.generate(chunk)
            audio_chunks.append(wav)

        print("Concatenating audio chunks...")
        wav = torch.cat(audio_chunks, dim=-1)

    # Save to bytes
    buffer = io.BytesIO()
    ta.save(buffer, wav, model.sr, format="wav")
    buffer.seek(0)
    return buffer.read()


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Handle TTS inference requests."""
    try:
        job_input = job["input"]
        text = job_input["text"]

        print("Processing ChatterboxTurboTTS request...")

        audio_bytes = synthesize_chatterbox(text)

        # Encode as base64
        audio_b64 = base64.b64encode(audio_bytes).decode()

        print("Speech synthesized successfully")

        return {"audio": audio_b64}

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Start the serverless worker
runpod.serverless.start({"handler": handler})
