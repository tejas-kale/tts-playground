"""ChatterboxTurboTTS model implementation using Runpod."""

from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from rich.console import Console

console = Console()


class ChatterboxModel:
    """ChatterboxTurboTTS model client for Runpod serverless inference."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint_id: str | None = None,
    ):
        """Initialize the Chatterbox client.

        Args:
            api_key: Runpod API key (or set RUNPOD_API_KEY env var)
            endpoint_id: Runpod endpoint ID (or set RUNPOD_ENDPOINT_ID env var)
        """
        self.api_key = api_key or os.getenv('RUNPOD_API_KEY')
        self.endpoint_id = endpoint_id or os.getenv('RUNPOD_ENDPOINT_ID')

        if not self.api_key or not self.endpoint_id:
            raise ValueError(
                "Runpod credentials required. Set RUNPOD_API_KEY and "
                "RUNPOD_ENDPOINT_ID environment variables or pass them as arguments."
            )

        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def count_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Simple approximation: ~0.75 tokens per word
        words = len(text.split())
        return int(words * 0.75)

    def split_text_into_chunks(
        self, text: str, max_tokens: int = 100
    ) -> list[str]:
        """Split text into chunks at sentence boundaries.

        This is optimized for ChatterboxTurboTTS which has a 100 token limit.

        Args:
            text: Input text to split
            max_tokens: Maximum tokens per chunk (default 100)

        Returns:
            List of text chunks
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If sentence is too long, split on commas
            if sentence_tokens > max_tokens:
                parts = re.split(r'(?<=,)\s+', sentence)
                for part in parts:
                    part_tokens = self.count_tokens(part)
                    if current_tokens + part_tokens > max_tokens:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = [part]
                            current_tokens = part_tokens
                        else:
                            # Even single part is too long, just add it
                            chunks.append(part)
                    else:
                        current_chunk.append(part)
                        current_tokens += part_tokens
            else:
                # Check if adding this sentence exceeds limit
                if current_tokens + sentence_tokens > max_tokens:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens

        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def synthesize_chunk(self, text: str) -> bytes:
        """Synthesize speech for a single text chunk via Runpod.

        Args:
            text: Text to synthesize (should be under 100 tokens)

        Returns:
            Audio data as bytes (WAV format)

        Raises:
            RuntimeError: If synthesis fails
        """
        # Submit job
        payload = {"input": {"text": text}}

        response = requests.post(
            f"{self.base_url}/run",
            json=payload,
            headers=self.headers,
            timeout=300,
        )
        response.raise_for_status()
        job_id = response.json()["id"]

        # Poll for completion until job reaches terminal state
        # Terminal states: COMPLETED, FAILED, CANCELLED, TIMED_OUT
        # Active states: IN_QUEUE, IN_PROGRESS
        max_wait_time = 1800  # 30 minutes safety timeout
        start_time = time.time()
        last_status_print = start_time
        status_print_interval = 60  # Print status every 60 seconds

        while True:
            # Check safety timeout
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                raise RuntimeError(
                    f"Job exceeded maximum wait time ({max_wait_time}s)"
                )

            response = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            status_data = response.json()

            status = status_data.get("status")

            if status == "COMPLETED":
                output = status_data.get("output", {})
                if "error" in output:
                    raise RuntimeError(f"Synthesis error: {output['error']}")

                audio_b64 = output.get("audio")
                if not audio_b64:
                    raise RuntimeError("No audio data in response")

                return base64.b64decode(audio_b64)

            elif status == "FAILED":
                error = status_data.get("error", "Unknown error")
                raise RuntimeError(f"Job failed: {error}")

            elif status == "CANCELLED":
                raise RuntimeError("Job was cancelled")

            elif status == "TIMED_OUT":
                raise RuntimeError("Job timed out on server")

            # Print periodic status updates
            current_time = time.time()
            if current_time - last_status_print >= status_print_interval:
                elapsed_mins = int(elapsed // 60)
                elapsed_secs = int(elapsed % 60)
                console.print(
                    f"[yellow]Job {status}: waiting {elapsed_mins}m {elapsed_secs}s "
                    f"(job_id: {job_id[:8]}...)[/yellow]"
                )
                last_status_print = current_time

            # Job still running (IN_QUEUE or IN_PROGRESS), wait and poll again
            time.sleep(5)

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
    ) -> Path:
        """Synthesize speech for text of any length.

        Automatically chunks text longer than 100 tokens.

        Args:
            text: Input text
            output_path: Path to save the audio file

        Returns:
            Path to the generated audio file
        """
        output_path = Path(output_path)

        # Count tokens
        total_tokens = self.count_tokens(text)

        if total_tokens <= 100:
            console.print(
                f"[cyan]Synthesizing {total_tokens} tokens with Chatterbox...[/cyan]"
            )
            audio_data = self.synthesize_chunk(text)

            # Save audio
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_data)

            console.print(f"[green]✓ Audio generated successfully[/green]")
            return output_path

        # Split into chunks
        console.print(
            f"[cyan]Text has {total_tokens} tokens (exceeds 100 token limit)[/cyan]"
        )
        chunks = self.split_text_into_chunks(text, max_tokens=100)
        console.print(f"[cyan]Split into {len(chunks)} chunks[/cyan]")

        # Synthesize each chunk
        audio_chunks = []
        for i, chunk in enumerate(chunks, 1):
            chunk_tokens = self.count_tokens(chunk)
            console.print(
                f"[cyan]  Chunk {i}/{len(chunks)}: {chunk_tokens} tokens[/cyan]"
            )

            audio_data = self.synthesize_chunk(chunk)
            audio_chunks.append(audio_data)

        # Concatenate audio chunks
        console.print("[cyan]Concatenating audio chunks...[/cyan]")

        # Lazy import torch and torchaudio (only needed for concatenation)
        import io

        import torch
        import torchaudio as ta

        # Load all chunks as tensors
        tensors = []
        sample_rate = None

        for audio_bytes in audio_chunks:
            buffer = io.BytesIO(audio_bytes)
            waveform, sr = ta.load(buffer)
            tensors.append(waveform)
            if sample_rate is None:
                sample_rate = sr

        # Concatenate
        concatenated = torch.cat(tensors, dim=-1)

        # Save final audio
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(output_path), concatenated, sample_rate)

        console.print(f"[green]✓ Audio generated successfully[/green]")
        return output_path
