"""VibeVoice model implementation using Runpod serverless."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import requests
from rich.console import Console

console = Console()


class VibeVoiceModel:
    """VibeVoice model client for Runpod serverless inference."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint_id: str | None = None,
        docker_image: str | None = None,
        auto_setup: bool = True,
    ):
        """Initialize the VibeVoice client.

        Args:
            api_key: Runpod API key (or set RUNPOD_API_KEY env var)
            endpoint_id: Runpod endpoint ID (or auto-created if auto_setup=True)
            docker_image: Docker image for VibeVoice deployment
            auto_setup: Automatically create template and endpoint if needed
        """
        self.api_key = api_key or os.getenv('RUNPOD_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Runpod API key required. Set RUNPOD_API_KEY environment "
                "variable or pass it as an argument."
            )

        self.docker_image = docker_image or os.getenv(
            'VIBEVOICE_DOCKER_IMAGE',
            'tejaskale/vibevoice-runpod:latest'
        )

        # Get or create endpoint
        if endpoint_id:
            self.endpoint_id = endpoint_id
        elif auto_setup:
            console.print("[cyan]Setting up VibeVoice endpoint...[/cyan]")
            self.endpoint_id = self._setup_endpoint()
        else:
            raise ValueError(
                "Either endpoint_id must be provided or auto_setup must be True"
            )

        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _setup_endpoint(self) -> str:
        """Setup template and endpoint automatically.

        Returns:
            Endpoint ID
        """
        from article_tts.runpod_manager import RunpodManager

        manager = RunpodManager(api_key=self.api_key)

        # Get GPU type from environment or use default
        gpu_ids = os.getenv('VIBEVOICE_GPU_TYPE', 'AMPERE_16')

        endpoint_id = manager.ensure_vibevoice_endpoint(
            docker_image=self.docker_image,
            gpu_ids=gpu_ids,
        )

        return endpoint_id

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        voice_sample_path: str | Path | None = None,
        add_speaker_labels: bool = True,
        chunk_size: int = 500,
    ) -> Path:
        """Synthesize speech using VibeVoice via Runpod.

        Args:
            text: Input text
            output_path: Path to save the audio file
            voice_sample_path: Path to voice sample (optional)
            add_speaker_labels: Whether to add speaker labels automatically
            chunk_size: Maximum characters per chunk for long text

        Returns:
            Path to the generated audio file
        """
        output_path = Path(output_path)

        console.print("[cyan]Synthesizing with VibeVoice (Runpod)...[/cyan]")

        # Prepare voice sample if provided
        voice_sample_b64 = None
        if voice_sample_path:
            voice_path = Path(voice_sample_path)
            if not voice_path.exists():
                console.print(f"[yellow]Warning: Voice sample not found: {voice_path}[/yellow]")
            else:
                console.print(f"[cyan]Using voice sample: {voice_path}[/cyan]")
                voice_sample_b64 = base64.b64encode(voice_path.read_bytes()).decode()

        # Prepare payload
        payload = {
            "input": {
                "text": text,
                "add_speaker_labels": add_speaker_labels,
                "chunk_size": chunk_size,
            }
        }

        if voice_sample_b64:
            payload["input"]["voice_sample"] = voice_sample_b64

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
