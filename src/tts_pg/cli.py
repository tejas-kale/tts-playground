"""Command-line interface for TTS Playground."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

from tts_pg import __version__
from tts_pg.config import Config
from tts_pg.models import ChatterboxModel, VibeVoiceModel
from tts_pg.runpod_manager import RunpodManager
from tts_pg.utils import (
    adjust_speed,
    convert_to_mp3,
    ensure_output_dir,
    play_audio,
    preprocess_text,
    read_text_file,
)

# Load environment variables
load_dotenv()

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="tts-pg")
def cli():
    """TTS Playground - Unified CLI for text-to-speech synthesis.

    Supports ChatterboxTurboTTS and VibeVoice via Runpod serverless.
    """
    pass


@cli.command()
@click.option(
    '--api-key',
    envvar='RUNPOD_API_KEY',
    help='Runpod API key',
    prompt='Runpod API key',
    hide_input=True,
)
@click.option(
    '--docker-image',
    default='runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04',
    help='Docker image for the unified endpoint',
)
@click.option(
    '--gpu-type',
    default='AMPERE_16',
    type=click.Choice(['AMPERE_16', 'AMPERE_24', 'AMPERE_48', 'ADA_24']),
    help='GPU type to use (default: AMPERE_16)',
)
@click.option(
    '--workers-max',
    default=1,
    type=int,
    help='Maximum number of workers (default: 1)',
)
@click.option(
    '--hf-token',
    envvar='HF_TOKEN',
    help='Hugging Face token (required for Chatterbox)',
)
@click.option(
    '--show-status/--no-show-status',
    default=False,
    help='Monitor endpoint deployment and startup status (blocking)',
)
def configure(
    api_key: str,
    docker_image: str,
    gpu_type: str,
    workers_max: int,
    hf_token: str | None,
    show_status: bool,
):
    """Configure TTS Playground by creating a unified Runpod endpoint.

    This command creates a Runpod template and deploys a unified endpoint
    that supports both ChatterboxTurboTTS and VibeVoice models.

    You only need to run this once. The endpoint ID will be saved locally
    and used for all future synthesis requests.

    Example:

        \b
        $ tts-pg configure
        Runpod API key: ****
        Hugging Face token (optional): ****
        ✓ Configuration complete!
    """
    try:
        console.print("\n[bold cyan]Configuring TTS Playground...[/bold cyan]\n")

        # Initialize Runpod manager
        manager = RunpodManager(api_key=api_key)

        # Create template
        console.print("[cyan]Creating Runpod template...[/cyan]")

        env_vars = {}
        if hf_token:
            env_vars['HF_TOKEN'] = hf_token

        # Create comprehensive start command
        docker_args = (
            "/bin/bash -c '"
            "apt-get update && "
            "apt-get install -y git ffmpeg libsndfile1 && "
            "pip install runpod chatterbox-tts soundfile pydub torchaudio accelerate && "
            "git clone https://github.com/vibevoice-community/VibeVoice.git /tmp/VibeVoice && "
            "cd /tmp/VibeVoice && pip install -e . && cd / && "
            "wget -O /handler.py https://raw.githubusercontent.com/tejas-kale/tts-playground/main/runpod_deployments/unified/handler.py && "
            "python /handler.py"
            "'"
        )

        template_name = "tts-pg-unified"
        template_id = manager.create_template(
            name=template_name,
            image_name=docker_image,
            docker_args=docker_args,
            container_disk_gb=20,  # More space for installations
            env_vars=env_vars,
            readme="Unified TTS Playground endpoint supporting ChatterboxTurboTTS and VibeVoice",
        )

        # Create endpoint
        console.print("[cyan]Creating Runpod endpoint...[/cyan]")

        endpoint_name = "tts-pg-unified-endpoint"
        endpoint_id = manager.create_endpoint(
            name=endpoint_name,
            template_id=template_id,
            gpu_ids=gpu_type,
            workers_min=0,
            workers_max=workers_max,
            idle_timeout=5,
        )

        # Save configuration
        config = Config()
        config.update({
            'api_key': api_key,
            'endpoint_id': endpoint_id,
            'template_id': template_id,
            'docker_image': docker_image,
            'gpu_type': gpu_type,
        })

        console.print(f"\n[bold green]✓ Configuration complete![/bold green]")
        console.print(f"[green]Endpoint ID: {endpoint_id}[/green]")
        console.print(f"[green]Config saved to: {config.config_file}[/green]")

        # Monitor deployment if requested
        if show_status:
            console.print("\n[cyan]Monitoring endpoint deployment...[/cyan]")
            manager.monitor_endpoint_startup(endpoint_id)
        else:
            console.print("\n[cyan]You can now use 'tts-pg speak' to generate audio![/cyan]\n")
            console.print("[yellow]Note: First request may take 3-5 minutes as the container installs dependencies.[/yellow]\n")

    except Exception as e:
        console.print(f"[red]Configuration failed: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.argument('text', required=False)
@click.option(
    '--file', '-f',
    type=click.Path(exists=True),
    help='Read text from file instead of argument'
)
@click.option(
    '--model', '-m',
    type=click.Choice(['chatterbox', 'vibevoice'], case_sensitive=False),
    default='chatterbox',
    help='TTS model to use (default: chatterbox)'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Output file path (default: /tmp/tts_pg_<timestamp>.wav)'
)
@click.option(
    '--format',
    type=click.Choice(['wav', 'mp3'], case_sensitive=False),
    default='wav',
    help='Output format (default: wav)'
)
@click.option(
    '--speed', '-s',
    type=float,
    default=0.85,
    help='Playback speed multiplier (default: 0.85)'
)
@click.option(
    '--bitrate', '-b',
    type=click.Choice(['128k', '192k', '320k']),
    default='192k',
    help='MP3 bitrate (default: 192k)'
)
@click.option(
    '--play/--no-play',
    default=True,
    help='Auto-play generated audio (default: enabled)'
)
@click.option(
    '--voice-sample',
    type=click.Path(exists=True),
    help='Voice sample for VibeVoice (optional)'
)
def speak(
    text: str | None,
    file: str | None,
    model: str,
    output: str | None,
    format: str,
    speed: float,
    bitrate: str,
    play: bool,
    voice_sample: str | None,
):
    """Generate speech from text using the specified model.

    Examples:

        \b
        # Use Chatterbox with text argument
        $ tts-pg speak "Hello, world!" -m chatterbox

        \b
        # Use VibeVoice with file input
        $ tts-pg speak -f article.txt -m vibevoice

        \b
        # Generate MP3 at custom speed
        $ tts-pg speak -f article.txt --format mp3 --speed 1.0

        \b
        # Use custom voice with VibeVoice
        $ tts-pg speak -f article.txt -m vibevoice --voice-sample voice.wav
    """
    # Validate input
    if not text and not file:
        raise click.UsageError("Either TEXT argument or --file option is required")

    if text and file:
        raise click.UsageError("Cannot specify both TEXT argument and --file option")

    # Get text content
    if file:
        try:
            text = read_text_file(file)
            console.print(f"[green]✓ Read text from {file}[/green]")
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            raise click.Abort()

    # Preprocess text
    text = preprocess_text(text)

    if not text.strip():
        console.print("[red]Error: Empty text provided[/red]")
        raise click.Abort()

    # Generate output filename if not specified
    if not output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f"/tmp/tts_pg_{timestamp}.wav"

    output_path = Path(output)
    ensure_output_dir(output_path)

    # Make sure output is WAV initially (we'll convert to MP3 later if needed)
    if format == 'mp3':
        wav_output = output_path.with_suffix('.wav')
    else:
        wav_output = output_path

    try:
        # Check if configured
        config = Config()
        if not config.is_configured():
            console.print(
                "[red]Error: TTS Playground is not configured.[/red]\n\n"
                "Please run 'tts-pg configure' first to set up your Runpod endpoint.\n\n"
                "Example:\n"
                "  $ tts-pg configure\n"
            )
            raise click.Abort()

        # Get configuration
        api_key = config.get_api_key()
        endpoint_id = config.get_endpoint_id()

        if not api_key or not endpoint_id:
            console.print(
                "[red]Error: Configuration is incomplete.[/red]\n"
                "Please run 'tts-pg configure' again."
            )
            raise click.Abort()

        # Synthesize speech based on model
        if model.lower() == 'chatterbox':
            console.print("[bold cyan]Using ChatterboxTurboTTS[/bold cyan]")

            tts_model = ChatterboxModel(
                api_key=api_key,
                endpoint_id=endpoint_id,
            )

            tts_model.synthesize(text, wav_output)

        elif model.lower() == 'vibevoice':
            console.print("[bold cyan]Using VibeVoice[/bold cyan]")

            tts_model = VibeVoiceModel(
                api_key=api_key,
                endpoint_id=endpoint_id,
            )

            tts_model.synthesize(
                text,
                wav_output,
                voice_sample_path=voice_sample,
            )

        # Apply speed adjustment if needed
        final_wav = wav_output
        if speed != 1.0:
            console.print(f"[cyan]Adjusting speed to {speed}x...[/cyan]")
            final_wav = adjust_speed(wav_output, speed)

        # Convert to MP3 if requested
        final_output = final_wav
        if format == 'mp3':
            console.print(f"[cyan]Converting to MP3 (bitrate: {bitrate})...[/cyan]")
            final_output = convert_to_mp3(final_wav, bitrate)

            # Clean up WAV files
            if Path(final_wav).exists() and final_wav != wav_output:
                Path(final_wav).unlink()
            if wav_output.exists() and format == 'mp3':
                wav_output.unlink()

        # Move to final output path if different
        if str(final_output) != str(output):
            final_path = Path(final_output)
            final_path.rename(output_path)
            final_output = output_path

        console.print(f"\n[bold green]✓ Audio saved to: {final_output}[/bold green]")

        # Auto-play if enabled
        if play:
            console.print("[cyan]Playing audio...[/cyan]")
            play_audio(final_output)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.argument('text', required=False)
@click.option(
    '--file', '-f',
    type=click.Path(exists=True),
    help='Read text from file'
)
def preprocess(text: str | None, file: str | None):
    """Preprocess and validate text for TTS.

    Useful for checking how text will be processed before synthesis.
    """
    # Validate input
    if not text and not file:
        raise click.UsageError("Either TEXT argument or --file option is required")

    # Get text content
    if file:
        text = read_text_file(file)

    # Preprocess
    processed = preprocess_text(text)

    # Display results
    console.print("\n[bold cyan]Original text:[/bold cyan]")
    console.print(text[:500] + "..." if len(text) > 500 else text)

    console.print("\n[bold cyan]Preprocessed text:[/bold cyan]")
    console.print(processed[:500] + "..." if len(processed) > 500 else processed)

    console.print(f"\n[bold]Length:[/bold] {len(processed)} characters")


@cli.command()
def info():
    """Display information about available models and configuration."""
    console.print("[bold cyan]TTS Playground - Model Information[/bold cyan]\n")

    console.print("[bold]Available Models:[/bold]")
    console.print("  • chatterbox - ChatterboxTurboTTS (fast, 10x speed)")
    console.print("  • vibevoice  - VibeVoice (voice cloning capable)\n")

    console.print("[bold]ChatterboxTurboTTS:[/bold]")
    console.print("  • Optimized for speed with 100 token chunks")
    console.print("  • Automatic text splitting for long content")
    console.print("  • Requires Hugging Face token\n")

    console.print("[bold]VibeVoice:[/bold]")
    console.print("  • Supports custom voice samples for voice cloning")
    console.print("  • Automatic chunking for long text (500 char chunks)")
    console.print("  • High-quality voice generation\n")

    console.print("[bold]Unified Endpoint:[/bold]")
    console.print("  • Both models run on a single Runpod endpoint")
    console.print("  • Set up once with 'tts-pg configure'")
    console.print("  • Switch between models with --model flag\n")

    # Check current configuration
    config = Config()

    if config.is_configured():
        console.print("[bold green]✓ Configuration Status: Configured[/bold green]")
        config.display()
    else:
        console.print("[bold yellow]✗ Configuration Status: Not Configured[/bold yellow]")
        console.print("\n[yellow]Run 'tts-pg configure' to set up your endpoint.[/yellow]")


if __name__ == '__main__':
    cli()
