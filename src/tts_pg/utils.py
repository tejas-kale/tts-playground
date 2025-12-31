"""Utility functions for text-to-speech processing."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path

from pydub import AudioSegment
from rich.console import Console

console = Console()


def preprocess_text(text: str) -> str:
    """Clean and format text for TTS processing.

    Args:
        text: Raw input text

    Returns:
        Cleaned text ready for TTS
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special formatting characters
    text = text.replace('\u200b', '')  # Zero-width space
    text = text.replace('\xa0', ' ')   # Non-breaking space

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    # Ensure proper sentence spacing
    text = re.sub(r'\.(?=[A-Z])', '. ', text)

    return text.strip()


def read_text_file(file_path: str | Path) -> str:
    """Read text from a file.

    Args:
        file_path: Path to the text file

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return path.read_text(encoding='utf-8')


def adjust_speed(audio_path: str | Path, speed: float) -> str:
    """Adjust audio playback speed.

    Args:
        audio_path: Path to the audio file
        speed: Speed multiplier (e.g., 0.85 for 85% speed)

    Returns:
        Path to the speed-adjusted audio file
    """
    if speed == 1.0:
        return str(audio_path)

    audio = AudioSegment.from_file(str(audio_path))

    # Calculate new frame rate
    new_frame_rate = int(audio.frame_rate * speed)

    # Apply speed change
    adjusted = audio._spawn(audio.raw_data, overrides={
        "frame_rate": new_frame_rate
    })
    adjusted = adjusted.set_frame_rate(audio.frame_rate)

    # Save adjusted audio
    output_path = Path(audio_path).with_suffix('.adjusted.wav')
    adjusted.export(str(output_path), format='wav')

    return str(output_path)


def convert_to_mp3(wav_path: str | Path, bitrate: str = '192k') -> str:
    """Convert WAV file to MP3.

    Args:
        wav_path: Path to the WAV file
        bitrate: MP3 bitrate (e.g., '128k', '192k', '320k')

    Returns:
        Path to the MP3 file
    """
    wav_path = Path(wav_path)
    mp3_path = wav_path.with_suffix('.mp3')

    audio = AudioSegment.from_wav(str(wav_path))
    audio.export(str(mp3_path), format='mp3', bitrate=bitrate)

    return str(mp3_path)


def play_audio(audio_path: str | Path) -> None:
    """Play audio file using system default player.

    Args:
        audio_path: Path to the audio file
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        console.print(f"[red]Error: Audio file not found: {audio_path}[/red]")
        return

    system = platform.system()

    try:
        if system == 'Darwin':  # macOS
            subprocess.run(['afplay', str(audio_path)], check=True)
        elif system == 'Linux':
            # Try multiple players
            players = ['paplay', 'aplay', 'mpg123', 'ffplay']
            for player in players:
                try:
                    subprocess.run([player, str(audio_path)],
                                 check=True,
                                 stderr=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL)
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
        elif system == 'Windows':
            os.startfile(str(audio_path))
        else:
            console.print(f"[yellow]Auto-play not supported on {system}[/yellow]")
            console.print(f"Audio saved to: {audio_path}")
    except Exception as e:
        console.print(f"[yellow]Could not auto-play audio: {e}[/yellow]")
        console.print(f"Audio saved to: {audio_path}")


def ensure_output_dir(output_path: str | Path) -> Path:
    """Ensure output directory exists.

    Args:
        output_path: Output file path

    Returns:
        Path object with created parent directory
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def count_tokens_simple(text: str) -> int:
    """Estimate token count (simple word-based approximation).

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Simple approximation: ~0.75 tokens per word
    words = len(text.split())
    return int(words * 0.75)
