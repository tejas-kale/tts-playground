# Article TTS

A unified CLI tool for high-quality text-to-speech synthesis supporting multiple models.

## Features

- **Two TTS Models**: ChatterboxTurboTTS (serverless via Runpod) and VibeVoice (local inference)
- **Intelligent Text Chunking**: Automatically splits long text at sentence boundaries
- **Speed Control**: Adjust playback speed (0.5x - 2.0x)
- **Format Options**: Output as WAV or MP3 with configurable bitrate
- **Auto-play**: Optional automatic audio playback after generation
- **Clean Interface**: Simple, intuitive CLI powered by Click

## Installation

Install using `uv`:

```bash
uv tool install article-tts
```

Or install from source:

```bash
git clone https://github.com/tejas-kale/tts-playground.git
cd tts-playground
uv tool install .
```

## Quick Start

### ChatterboxTurboTTS (Serverless)

1. Set up Runpod credentials:

```bash
export RUNPOD_API_KEY="your-api-key"
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
```

2. Generate speech:

```bash
# From text argument
article-tts speak "Hello, world!" -m chatterbox

# From file
article-tts speak -f article.txt -m chatterbox

# Custom output location and format
article-tts speak -f article.txt -m chatterbox -o output.mp3 --format mp3
```

### VibeVoice (Local)

1. Install VibeVoice:

```bash
git clone https://github.com/tejas-kale/VibeVoice.git
cd VibeVoice && pip install -e .
```

2. Generate speech:

```bash
# Basic usage
article-tts speak "Hello, world!" -m vibevoice

# With custom voice sample
article-tts speak -f article.txt -m vibevoice --voice-sample voice.wav

# Adjust speed and format
article-tts speak -f article.txt -m vibevoice --speed 1.0 --format mp3
```

## Usage

### Basic Commands

```bash
# Generate speech
article-tts speak [TEXT] [OPTIONS]

# Preprocess and validate text
article-tts preprocess [TEXT] [OPTIONS]

# Display model information
article-tts info
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model, -m` | Model to use (`chatterbox` or `vibevoice`) | `chatterbox` |
| `--file, -f` | Read text from file | - |
| `--output, -o` | Output file path | `/tmp/article_tts_<timestamp>.wav` |
| `--format` | Output format (`wav` or `mp3`) | `wav` |
| `--speed, -s` | Playback speed multiplier | `0.85` |
| `--bitrate, -b` | MP3 bitrate (`128k`, `192k`, `320k`) | `192k` |
| `--play/--no-play` | Auto-play generated audio | `enabled` |
| `--voice-sample` | Voice sample for VibeVoice | - |

### Examples

```bash
# Generate MP3 at normal speed
article-tts speak -f article.txt --format mp3 --speed 1.0

# Use custom voice with VibeVoice
article-tts speak "Hello!" -m vibevoice --voice-sample my_voice.wav

# Save to custom location without auto-play
article-tts speak -f book.txt -o ~/audiobooks/chapter1.wav --no-play

# Preprocess text to see what will be synthesized
article-tts preprocess -f article.txt

# Check model availability and configuration
article-tts info
```

## Model Comparison

### ChatterboxTurboTTS

**Advantages:**
- Fast inference via Runpod serverless
- No local GPU required
- Optimized for speed (10x faster than standard models)
- Automatic chunking at 100 token boundaries

**Requirements:**
- Runpod account with API key
- Configured Runpod endpoint with ChatterboxTurboTTS

**Optimizations:**
- Text split at sentence boundaries with 100 token limit
- Automatic audio concatenation for seamless output
- Efficient serverless execution

### VibeVoice

**Advantages:**
- Custom voice cloning with voice samples
- Local inference for privacy
- High-quality voice generation
- Flexible speaker labeling

**Requirements:**
- Local GPU (CUDA recommended)
- VibeVoice library installed

**Optimizations:**
- Intelligent chunking for long text (500 character chunks)
- Automatic speaker label injection
- Optional text preprocessing with GPT models
- Seamless audio concatenation

## Configuration

### Environment Variables

Create a `.env` file in your working directory:

```bash
# For ChatterboxTurboTTS
RUNPOD_API_KEY=your-api-key-here
RUNPOD_ENDPOINT_ID=your-endpoint-id-here
```

Or set them in your shell:

```bash
export RUNPOD_API_KEY="your-api-key"
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
```

### Voice Samples (VibeVoice)

For best results with VibeVoice:
- Use clean audio samples (3-10 seconds)
- WAV format recommended
- Clear speech without background noise
- Sample rate of 16kHz or higher

## Architecture

```
article-tts/
├── src/article_tts/
│   ├── __init__.py          # Package initialization
│   ├── cli.py               # CLI interface
│   ├── utils.py             # Shared utilities
│   └── models/
│       ├── __init__.py      # Model exports
│       ├── chatterbox.py    # ChatterboxTurboTTS implementation
│       └── vibevoice.py     # VibeVoice implementation
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Development

### Install for Development

```bash
git clone https://github.com/tejas-kale/tts-playground.git
cd tts-playground

# Install with uv
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Code Formatting

```bash
black src/
ruff check src/
```

## Output Location

By default, audio files are saved to `/tmp` with timestamps:
```
/tmp/article_tts_20250130_143022.wav
```

Specify a custom output location with `-o`:
```bash
article-tts speak -f article.txt -o ~/audio/output.wav
```

## Troubleshooting

### ChatterboxTurboTTS Issues

**"Runpod credentials required"**
- Ensure `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` are set
- Verify credentials are correct

**"Job timed out"**
- Check Runpod endpoint status
- Ensure endpoint has workers available
- Try shorter text

### VibeVoice Issues

**"VibeVoice is not installed"**
- Install VibeVoice: `git clone https://github.com/tejas-kale/VibeVoice.git && cd VibeVoice && pip install -e .`

**"CUDA out of memory"**
- Use shorter text or enable chunking
- Reduce batch size in model configuration
- Try CPU inference (slower): set device='cpu'

**Audio quality issues with voice samples**
- Use higher quality voice samples (16kHz+)
- Ensure sample is clean (no background noise)
- Try samples between 3-10 seconds

## License

MIT License - See LICENSE file for details

## Credits

- **ChatterboxTurboTTS**: Based on the Chatterbox model
- **VibeVoice**: Based on the VibeVoice project by Tejas Kale

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Links

- [GitHub Repository](https://github.com/tejas-kale/tts-playground)
- [Issue Tracker](https://github.com/tejas-kale/tts-playground/issues)
- [VibeVoice Repository](https://github.com/tejas-kale/VibeVoice)
