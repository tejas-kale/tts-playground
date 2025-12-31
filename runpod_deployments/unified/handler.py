"""Unified Runpod handler for ChatterboxTurboTTS and VibeVoice.

This handler supports both TTS models in a single endpoint.
"""

from __future__ import annotations

import base64
import io
import os
import re
from typing import Any

import runpod
import soundfile as sf
import torch
import torchaudio as ta
from pydub import AudioSegment


# Global models (loaded once on cold start)
chatterbox_model = None
vibevoice_model = None
vibevoice_processor = None


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


def load_vibevoice():
    """Load the VibeVoice model."""
    global vibevoice_model, vibevoice_processor

    if vibevoice_model is None:
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        model_name = os.getenv("VIBEVOICE_MODEL", "microsoft/VibeVoice-1.5B")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Loading VibeVoice model '{model_name}' on {device}...")
        vibevoice_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_name
        ).to(device)
        vibevoice_processor = VibeVoiceProcessor.from_pretrained(model_name)
        print("VibeVoice model loaded successfully!")

    return vibevoice_model, vibevoice_processor


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


def preprocess_text_vibevoice(text: str, add_speaker: bool = True) -> str:
    """Preprocess text for VibeVoice."""
    text = text.strip()

    if add_speaker:
        lines = text.split('\n')
        processed_lines = []
        for line in lines:
            line = line.strip()
            if line and not re.match(r'^Speaker \d+:', line):
                line = f"Speaker 1: {line}"
            processed_lines.append(line)
        text = '\n'.join(processed_lines)

    return text


def synthesize_vibevoice_chunk(
    text: str,
    voice_sample_b64: str | None = None,
    model=None,
    processor=None,
    max_length: int = 2000,
) -> bytes:
    """Synthesize a chunk using VibeVoice."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare inputs
    if voice_sample_b64:
        voice_bytes = base64.b64decode(voice_sample_b64)
        voice_buffer = io.BytesIO(voice_bytes)
        voice_audio, voice_sr = sf.read(voice_buffer)

        inputs = processor(
            text=text,
            audio=voice_audio,
            sampling_rate=voice_sr,
            return_tensors="pt"
        )
    else:
        inputs = processor(text=text, return_tensors="pt")

    # Move to device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v
             for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        output = model.generate(**inputs, tokenizer=processor.tokenizer, max_length=max_length)

    # Convert to numpy
    if torch.is_tensor(output):
        audio_array = output.cpu().numpy()
    else:
        audio_array = output

    # Get sample rate
    sample_rate = getattr(model.config, 'sample_rate', 24000)

    # Save to bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio_array.squeeze(), sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.read()


def synthesize_vibevoice(
    text: str,
    voice_sample_b64: str | None = None,
    add_speaker_labels: bool = True,
    chunk_size: int = 500,
) -> bytes:
    """Synthesize speech using VibeVoice."""
    model, processor = load_vibevoice()

    # Preprocess text
    processed_text = preprocess_text_vibevoice(text, add_speaker=add_speaker_labels)

    # Check if chunking needed
    if len(processed_text) > chunk_size:
        print(f"Text length {len(processed_text)} exceeds chunk size {chunk_size}")

        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', processed_text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        print(f"Split into {len(chunks)} chunks")

        # Synthesize each chunk
        audio_chunks = []
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}/{len(chunks)}")
            audio_bytes = synthesize_vibevoice_chunk(
                chunk, voice_sample_b64, model, processor
            )
            audio_chunks.append(audio_bytes)

        # Concatenate
        print("Concatenating audio chunks...")
        combined = AudioSegment.empty()
        for audio_bytes in audio_chunks:
            chunk_audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
            combined += chunk_audio

        # Export
        buffer = io.BytesIO()
        combined.export(buffer, format='wav')
        buffer.seek(0)
        return buffer.read()
    else:
        # Direct synthesis
        return synthesize_vibevoice_chunk(
            processed_text, voice_sample_b64, model, processor
        )


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Handle TTS inference requests for both models."""
    try:
        job_input = job["input"]

        # Get model selection
        model_name = job_input.get("model", "chatterbox").lower()
        text = job_input["text"]

        print(f"Processing request for model: {model_name}")

        if model_name == "chatterbox":
            audio_bytes = synthesize_chatterbox(text)
        elif model_name == "vibevoice":
            voice_sample_b64 = job_input.get("voice_sample")
            add_speaker_labels = job_input.get("add_speaker_labels", True)
            chunk_size = job_input.get("chunk_size", 500)

            audio_bytes = synthesize_vibevoice(
                text, voice_sample_b64, add_speaker_labels, chunk_size
            )
        else:
            return {"error": f"Unknown model: {model_name}"}

        # Encode as base64
        audio_b64 = base64.b64encode(audio_bytes).decode()

        print(f"Speech synthesized successfully with {model_name}")

        return {"audio": audio_b64}

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Start the serverless worker
runpod.serverless.start({"handler": handler})
