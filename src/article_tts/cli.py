"""Command-line interface for Article TTS."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

from article_tts import __version__
from article_tts.models import ChatterboxModel, VibeVoiceModel
from article_tts.utils import (
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
@click.version_option(version=__version__, prog_name="article-tts")
def cli():
    """Article TTS - Unified CLI for text-to-speech synthesis.

    Supports ChatterboxTurboTTS (via Runpod) and VibeVoice (local inference).
    """
    pass


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
    help='Output file path (default: /tmp/article_tts_<timestamp>.wav)'
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
@click.option(
    '--runpod-api-key',
    envvar='RUNPOD_API_KEY',
    help='Runpod API key (for Chatterbox)'
)
@click.option(
    '--runpod-endpoint-id',
    envvar='RUNPOD_ENDPOINT_ID',
    help='Runpod endpoint ID (for Chatterbox)'
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
    runpod_api_key: str | None,
    runpod_endpoint_id: str | None,
):
    """Generate speech from text using the specified model.

    Examples:

        \b
        # Use Chatterbox with text argument
        $ article-tts speak "Hello, world!" -m chatterbox

        \b
        # Use VibeVoice with file input
        $ article-tts speak -f article.txt -m vibevoice

        \b
        # Generate MP3 at custom speed
        $ article-tts speak -f article.txt --format mp3 --speed 1.0

        \b
        # Use custom voice with VibeVoice
        $ article-tts speak -f article.txt -m vibevoice --voice-sample voice.wav
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
        output = f"/tmp/article_tts_{timestamp}.wav"

    output_path = Path(output)
    ensure_output_dir(output_path)

    # Make sure output is WAV initially (we'll convert to MP3 later if needed)
    if format == 'mp3':
        wav_output = output_path.with_suffix('.wav')
    else:
        wav_output = output_path

    try:
        # Synthesize speech based on model
        if model.lower() == 'chatterbox':
            console.print("[bold cyan]Using ChatterboxTurboTTS[/bold cyan]")

            if not runpod_api_key or not runpod_endpoint_id:
                console.print(
                    "[red]Error: Runpod credentials required for Chatterbox.[/red]\n"
                    "Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID environment variables "
                    "or pass them as options."
                )
                raise click.Abort()

            chatterbox = ChatterboxModel(
                api_key=runpod_api_key,
                endpoint_id=runpod_endpoint_id,
            )

            chatterbox.synthesize(text, wav_output)

        elif model.lower() == 'vibevoice':
            console.print("[bold cyan]Using VibeVoice[/bold cyan]")

            try:
                vibevoice = VibeVoiceModel(voice_sample_path=voice_sample)

                # Determine if we need to use long text synthesis
                if len(text) > 1000:  # Use chunking for long text
                    vibevoice.synthesize_long_text(
                        text,
                        wav_output,
                        voice_sample_path=voice_sample,
                    )
                else:
                    vibevoice.synthesize(
                        text,
                        wav_output,
                        voice_sample_path=voice_sample,
                    )
            except ImportError as e:
                console.print(f"[red]Error: {e}[/red]")
                raise click.Abort()

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
    console.print("[bold cyan]Article TTS - Model Information[/bold cyan]\n")

    console.print("[bold]Available Models:[/bold]")
    console.print("  • chatterbox - ChatterboxTurboTTS via Runpod (serverless)")
    console.print("  • vibevoice  - VibeVoice local inference (requires GPU)\n")

    console.print("[bold]ChatterboxTurboTTS:[/bold]")
    console.print("  • Requires: RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID")
    console.print("  • Optimized for speed with 100 token chunks")
    console.print("  • Automatic text splitting for long content")
    console.print("  • Serverless inference via Runpod\n")

    console.print("[bold]VibeVoice:[/bold]")
    console.print("  • Requires: Local GPU (CUDA recommended)")
    console.print("  • Supports custom voice samples")
    console.print("  • Automatic chunking for long text")
    console.print("  • Local inference for privacy\n")

    console.print("[bold]Environment Variables:[/bold]")
    console.print("  • RUNPOD_API_KEY - Runpod API key (for Chatterbox)")
    console.print("  • RUNPOD_ENDPOINT_ID - Runpod endpoint ID (for Chatterbox)")

    # Check current configuration
    console.print("\n[bold]Current Configuration:[/bold]")
    runpod_key = os.getenv('RUNPOD_API_KEY')
    runpod_endpoint = os.getenv('RUNPOD_ENDPOINT_ID')

    console.print(
        f"  • Runpod API Key: {'✓ Set' if runpod_key else '✗ Not set'}"
    )
    console.print(
        f"  • Runpod Endpoint: {'✓ Set' if runpod_endpoint else '✗ Not set'}"
    )

    # Check for GPU
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        console.print(f"  • CUDA Available: {'✓ Yes' if has_cuda else '✗ No'}")
    except ImportError:
        console.print("  • CUDA Available: ✗ PyTorch not installed")


if __name__ == '__main__':
    cli()
