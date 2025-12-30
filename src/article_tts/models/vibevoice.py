"""VibeVoice model implementation for local inference."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from pydub import AudioSegment
from rich.console import Console

console = Console()


class VibeVoiceModel:
    """VibeVoice model for local text-to-speech inference."""

    def __init__(
        self,
        model_name: str = "seshurajup/VibeVoice",
        voice_sample_path: str | None = None,
        device: str | None = None,
    ):
        """Initialize the VibeVoice model.

        Args:
            model_name: Hugging Face model name
            voice_sample_path: Path to voice sample audio file (optional)
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        try:
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        except ImportError:
            raise ImportError(
                "VibeVoice is not installed. Install it with:\n"
                "  git clone https://github.com/tejas-kale/VibeVoice.git\n"
                "  cd VibeVoice && pip install -e ."
            )

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        console.print(f"[cyan]Loading VibeVoice model on {self.device}...[/cyan]")

        # Load model and processor
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_name
        ).to(self.device)
        self.processor = VibeVoiceProcessor.from_pretrained(model_name)

        self.voice_sample_path = voice_sample_path

        console.print("[green]✓ Model loaded successfully[/green]")

    def preprocess_text(self, text: str, add_speaker: bool = True) -> str:
        """Preprocess text for VibeVoice synthesis.

        Args:
            text: Input text
            add_speaker: Whether to add speaker labels if not present

        Returns:
            Preprocessed text
        """
        # Basic preprocessing
        text = text.strip()

        # Add speaker labels if requested and not already present
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

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        voice_sample_path: str | Path | None = None,
        add_speaker_labels: bool = True,
        max_length: int = 2000,
    ) -> Path:
        """Synthesize speech using VibeVoice.

        Args:
            text: Input text
            output_path: Path to save the audio file
            voice_sample_path: Path to voice sample (overrides instance setting)
            add_speaker_labels: Whether to automatically add speaker labels
            max_length: Maximum generation length

        Returns:
            Path to the generated audio file
        """
        output_path = Path(output_path)

        # Preprocess text
        processed_text = self.preprocess_text(text, add_speaker=add_speaker_labels)

        console.print("[cyan]Synthesizing with VibeVoice...[/cyan]")

        # Prepare inputs
        if voice_sample_path or self.voice_sample_path:
            sample_path = voice_sample_path or self.voice_sample_path
            console.print(f"[cyan]Using voice sample: {sample_path}[/cyan]")

            # Load voice sample
            voice_audio, voice_sr = sf.read(str(sample_path))

            # Process inputs with voice sample
            inputs = self.processor(
                text=processed_text,
                audio=voice_audio,
                sampling_rate=voice_sr,
                return_tensors="pt"
            )
        else:
            # Process without voice sample (uses default voice)
            console.print("[yellow]No voice sample provided, using default voice[/yellow]")
            inputs = self.processor(
                text=processed_text,
                return_tensors="pt"
            )

        # Move inputs to device
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v
                 for k, v in inputs.items()}

        # Generate audio
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=max_length,
            )

        # Convert output to numpy array
        if torch.is_tensor(output):
            audio_array = output.cpu().numpy()
        else:
            audio_array = output

        # Get sample rate from model config
        sample_rate = getattr(
            self.model.config, 'sample_rate', 24000
        )  # Default to 24kHz

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save audio
        sf.write(
            str(output_path),
            audio_array.squeeze(),
            sample_rate,
            format='WAV'
        )

        console.print(f"[green]✓ Audio generated successfully[/green]")
        return output_path

    def synthesize_long_text(
        self,
        text: str,
        output_path: str | Path,
        voice_sample_path: str | Path | None = None,
        chunk_size: int = 500,  # characters, not tokens
        add_speaker_labels: bool = True,
    ) -> Path:
        """Synthesize long text by splitting into chunks.

        Args:
            text: Input text
            output_path: Path to save the audio file
            voice_sample_path: Path to voice sample
            chunk_size: Maximum characters per chunk
            add_speaker_labels: Whether to add speaker labels

        Returns:
            Path to the generated audio file
        """
        output_path = Path(output_path)

        # Split text into chunks at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

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

        console.print(f"[cyan]Split text into {len(chunks)} chunks[/cyan]")

        # Synthesize each chunk
        audio_chunks = []
        temp_dir = output_path.parent / '.temp_chunks'
        temp_dir.mkdir(exist_ok=True)

        try:
            for i, chunk in enumerate(chunks, 1):
                console.print(f"[cyan]  Chunk {i}/{len(chunks)}[/cyan]")

                chunk_path = temp_dir / f'chunk_{i}.wav'
                self.synthesize(
                    chunk,
                    chunk_path,
                    voice_sample_path=voice_sample_path,
                    add_speaker_labels=add_speaker_labels,
                )

                # Load chunk audio
                chunk_audio = AudioSegment.from_wav(str(chunk_path))
                audio_chunks.append(chunk_audio)

            # Concatenate all chunks
            console.print("[cyan]Concatenating audio chunks...[/cyan]")
            combined = AudioSegment.empty()
            for chunk_audio in audio_chunks:
                combined += chunk_audio

            # Export final audio
            combined.export(str(output_path), format='wav')

            console.print(f"[green]✓ Long audio generated successfully[/green]")

        finally:
            # Clean up temp files
            for temp_file in temp_dir.glob('*.wav'):
                temp_file.unlink()
            temp_dir.rmdir()

        return output_path
