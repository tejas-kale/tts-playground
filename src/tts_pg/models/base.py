"""Base model class for unified Runpod TTS."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import requests
from rich.console import Console

console = Console()


class UnifiedTTSModel:
    """Unified TTS model client for Runpod serverless inference."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        endpoint_id: str | None = None,
    ):
        """Initialize the TTS client.

        Args:
            model_name: Model to use ('chatterbox' or 'vibevoice')
            api_key: Runpod API key (or set RUNPOD_API_KEY env var)
            endpoint_id: Runpod endpoint ID
        """
        self.model_name = model_name.lower()
        self.api_key = api_key or os.getenv('RUNPOD_API_KEY')
        self.endpoint_id = endpoint_id

        if not self.api_key:
            raise ValueError(
                "Runpod API key required. Set RUNPOD_API_KEY environment "
                "variable or pass it as an argument."
            )

        if not self.endpoint_id:
            raise ValueError(
                "Runpod endpoint ID required. Run 'tts-pg configure' "
                "to set up your endpoint."
            )

        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        **kwargs,
    ) -> Path:
        """Synthesize speech.

        Args:
            text: Input text
            output_path: Path to save the audio file
            **kwargs: Additional model-specific parameters

        Returns:
            Path to the generated audio file
        """
        output_path = Path(output_path)

        console.print(f"[cyan]Synthesizing with {self.model_name.title()}...[/cyan]")

        # Prepare payload
        payload = {
            "input": {
                "model": self.model_name,
                "text": text,
                **kwargs,
            }
        }

        # Submit job
        response = requests.post(
            f"{self.base_url}/run",
            json=payload,
            headers=self.headers,
            timeout=300,
        )
        response.raise_for_status()
        job_id = response.json()["id"]

        console.print(f"[cyan]Job submitted (ID: {job_id}), waiting for completion...[/cyan]")

        # Poll for completion
        max_attempts = 120  # 10 minutes max
        attempt = 0

        while attempt < max_attempts:
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

                # Decode and save audio
                audio_data = base64.b64decode(audio_b64)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(audio_data)

                console.print(f"[green]✓ Audio generated successfully[/green]")
                return output_path

            elif status == "FAILED":
                error = status_data.get("error", "Unknown error")
                raise RuntimeError(f"Job failed: {error}")

            # Wait before next poll
            time.sleep(5)
            attempt += 1

        raise RuntimeError("Job timed out after 10 minutes")


class ChatterboxModel(UnifiedTTSModel):
    """ChatterboxTurboTTS model client."""

    def __init__(self, api_key: str | None = None, endpoint_id: str | None = None):
        """Initialize Chatterbox client."""
        super().__init__("chatterbox", api_key=api_key, endpoint_id=endpoint_id)


class VibeVoiceModel(UnifiedTTSModel):
    """VibeVoice model client."""

    def __init__(self, api_key: str | None = None, endpoint_id: str | None = None):
        """Initialize VibeVoice client."""
        super().__init__("vibevoice", api_key=api_key, endpoint_id=endpoint_id)

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        voice_sample_path: str | Path | None = None,
        add_speaker_labels: bool = True,
        chunk_size: int = 500,
    ) -> Path:
        """Synthesize speech with VibeVoice-specific options."""
        kwargs = {
            "add_speaker_labels": add_speaker_labels,
            "chunk_size": chunk_size,
        }

        # Encode voice sample if provided
        if voice_sample_path:
            voice_path = Path(voice_sample_path)
            if not voice_path.exists():
                console.print(f"[yellow]Warning: Voice sample not found: {voice_path}[/yellow]")
            else:
                console.print(f"[cyan]Using voice sample: {voice_path}[/cyan]")
                voice_sample_b64 = base64.b64encode(voice_path.read_bytes()).decode()
                kwargs["voice_sample"] = voice_sample_b64

        return super().synthesize(text, output_path, **kwargs)
