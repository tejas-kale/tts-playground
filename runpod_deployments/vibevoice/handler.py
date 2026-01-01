"""Runpod handler for VibeVoice TTS inference.

This script runs on the Runpod instance and handles VibeVoice TTS inference requests.
"""

from __future__ import annotations

import base64
import io
import os
import re
import tempfile
from typing import Any

import runpod
import soundfile as sf
import torch
from pydub import AudioSegment


# Global model (loaded once on cold start)
model = None
processor = None
default_voice = None


def load_model():
    """Load the VibeVoice model.

    Returns:
        Tuple of (model, processor, default_voice)
    """
    global model, processor, default_voice

    if model is None:
        try:
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
            import urllib.request
        except ImportError:
            raise ImportError(
                "VibeVoice is not installed. This should not happen in the "
                "container as it's installed during build."
            )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = os.getenv("VIBEVOICE_MODEL", "microsoft/VibeVoice-1.5B")

        print(f"Loading VibeVoice model '{model_name}' on {device}...")

        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_name
        ).to(device)
        processor = VibeVoiceProcessor.from_pretrained(model_name)

        print("Model loaded successfully!")

        # Download default voice sample
        print("Downloading default voice sample...")
        voice_url = "https://raw.githubusercontent.com/vibevoice-community/VibeVoice/main/demo/voices/en-Alice_woman.wav"
        default_voice = "/tmp/vibevoice_default_voice.wav"
        urllib.request.urlretrieve(voice_url, default_voice)
        print(f"Default voice sample saved to {default_voice}")

    return model, processor, default_voice


def preprocess_text(text: str, add_speaker: bool = True) -> str:
    """Preprocess text for VibeVoice synthesis.

    Args:
        text: Input text
        add_speaker: Whether to add speaker labels if not present

    Returns:
        Preprocessed text
    """
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


def synthesize_chunk(
    text: str,
    voice_sample_b64: str | None = None,
    default_voice_path: str | None = None,
    tts_model=None,
    tts_processor=None,
    max_length: int = 2000,
) -> bytes:
    """Synthesize a single chunk of text.

    Args:
        text: Text to synthesize
        voice_sample_b64: Base64-encoded voice sample audio (optional)
        default_voice_path: Path to default voice sample
        tts_model: VibeVoice model
        tts_processor: VibeVoice processor
        max_length: Maximum generation length

    Returns:
        Audio data as bytes (WAV format)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare voice sample path
    if voice_sample_b64:
        # Save custom voice sample to temp file
        voice_bytes = base64.b64decode(voice_sample_b64)
        temp_voice = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_voice.write(voice_bytes)
        temp_voice.close()
        voice_sample_path = temp_voice.name
    else:
        # Use default voice
        voice_sample_path = default_voice_path

    # Prepare inputs - VibeVoice 1.5B requires text and voice_samples as lists
    inputs = tts_processor(
        text=[text],
        voice_samples=[[voice_sample_path]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Move to device
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    # Generate audio
    with torch.no_grad():
        outputs = tts_model.generate(
            **inputs,
            tokenizer=tts_processor.tokenizer,
            max_new_tokens=None,
            cfg_scale=1.3,
            generation_config={'do_sample': False},
        )

    # Clean up temp file if created
    if voice_sample_b64 and os.path.exists(voice_sample_path):
        os.unlink(voice_sample_path)

    # Extract audio from outputs
    if hasattr(outputs, 'speech_outputs') and outputs.speech_outputs:
        audio_array = outputs.speech_outputs[0]
    else:
        raise ValueError("No audio output generated")

    # Get sample rate
    sample_rate = 24000  # VibeVoice uses 24kHz

    # Save to bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio_array.squeeze(), sample_rate, format='WAV')
    buffer.seek(0)

    return buffer.read()


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Handle TTS inference requests.

    Args:
        job: Job input containing text and parameters

    Returns:
        Dictionary with generated audio
    """
    try:
        job_input = job["input"]

        # Get parameters
        text = job_input["text"]
        voice_sample_b64 = job_input.get("voice_sample")  # Optional
        add_speaker_labels = job_input.get("add_speaker_labels", True)
        chunk_size = job_input.get("chunk_size", 500)  # Characters

        # Load model
        tts_model, tts_processor, default_voice = load_model()

        # Preprocess text
        processed_text = preprocess_text(text, add_speaker=add_speaker_labels)

        print(f"Synthesizing speech for text: {processed_text[:50]}...")

        # Check if text is long and needs chunking
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
                audio_bytes = synthesize_chunk(
                    chunk,
                    voice_sample_b64,
                    default_voice,
                    tts_model,
                    tts_processor,
                )
                audio_chunks.append(audio_bytes)

            # Concatenate audio
            print("Concatenating audio chunks...")
            combined = AudioSegment.empty()
            for audio_bytes in audio_chunks:
                chunk_audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
                combined += chunk_audio

            # Export to bytes
            buffer = io.BytesIO()
            combined.export(buffer, format='wav')
            buffer.seek(0)
            final_audio = buffer.read()

        else:
            # Direct synthesis for short text
            final_audio = synthesize_chunk(
                processed_text,
                voice_sample_b64,
                default_voice,
                tts_model,
                tts_processor,
            )

        # Encode as base64
        audio_b64 = base64.b64encode(final_audio).decode()

        print("Speech synthesized successfully")

        return {"audio": audio_b64}

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Start the serverless worker
runpod.serverless.start({"handler": handler})
