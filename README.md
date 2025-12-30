# Article TTS

A unified CLI tool for high-quality text-to-speech synthesis supporting ChatterboxTurboTTS and VibeVoice via Runpod serverless.

## Features

- **Two TTS Models**: ChatterboxTurboTTS and VibeVoice, both running on Runpod serverless
- **Zero Local Setup**: No GPU or heavy dependencies required
- **Automatic Deployment**: VibeVoice endpoint created automatically on first use
- **Voice Cloning**: Support for custom voice samples with VibeVoice
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

### Prerequisites

1. **Runpod Account**: Sign up at [runpod.io](https://runpod.io)
2. **Get API Key**: Find it in Settings → API Keys
3. **Set Environment Variables**:

```bash
export RUNPOD_API_KEY="your-api-key"
```

### ChatterboxTurboTTS

For Chatterbox, you also need an endpoint ID:

```bash
export RUNPOD_ENDPOINT_ID="your-chatterbox-endpoint-id"
```

Then generate speech:

```bash
# From text argument
article-tts speak "Hello, world!" -m chatterbox

# From file
article-tts speak -f article.txt -m chatterbox

# Custom output and format
article-tts speak -f article.txt -o output.mp3 --format mp3
```

### VibeVoice (Automatic Setup)

VibeVoice automatically creates and deploys the endpoint on first use:

```bash
# Basic usage (endpoint auto-created)
article-tts speak "Hello, world!" -m vibevoice

# With custom voice sample
article-tts speak -f article.txt -m vibevoice --voice-sample voice.wav

# Adjust speed and format
article-tts speak -f article.txt -m vibevoice --speed 1.0 --format mp3
```

The first run will:
1. Create a template named "article-tts-vibevoice"
2. Deploy an endpoint named "article-tts-vibevoice-endpoint"
3. Start synthesis once ready

Subsequent runs use the existing endpoint instantly.

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
| `--voice-sample` | Voice sample for VibeVoice (optional) | - |
| `--runpod-api-key` | Runpod API key | `$RUNPOD_API_KEY` |
| `--runpod-endpoint-id` | Runpod endpoint ID (Chatterbox) | `$RUNPOD_ENDPOINT_ID` |

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
- Extremely fast inference (10x faster than standard)
- Optimized for speed and efficiency
- Requires pre-configured endpoint

**Requirements:**
- `RUNPOD_API_KEY`
- `RUNPOD_ENDPOINT_ID` (your Chatterbox endpoint)

**Optimizations:**
- Text split at sentence boundaries with 100 token limit
- Automatic audio concatenation for seamless output
- Efficient serverless execution

### VibeVoice

**Advantages:**
- Custom voice cloning with voice samples
- High-quality voice generation
- Automatic endpoint deployment
- No manual setup required

**Requirements:**
- `RUNPOD_API_KEY` only

**Optimizations:**
- Intelligent chunking for long text (500 character chunks)
- Automatic speaker label injection
- Seamless audio concatenation
- Auto-deployment on first use

## Configuration

### Environment Variables

Create a `.env` file in your working directory:

```bash
# Required for both models
RUNPOD_API_KEY=your-api-key-here

# Required for Chatterbox only
RUNPOD_ENDPOINT_ID=your-chatterbox-endpoint-id

# Optional for VibeVoice
VIBEVOICE_DOCKER_IMAGE=tejaskale/vibevoice-runpod:latest
VIBEVOICE_GPU_TYPE=AMPERE_16
```

Or set them in your shell:

```bash
export RUNPOD_API_KEY="your-api-key"
export RUNPOD_ENDPOINT_ID="your-endpoint-id"  # Chatterbox only
```

### GPU Types

For VibeVoice, you can customize the GPU type:

- `AMPERE_16` - RTX 3060, A2000, A4000 (default, cost-effective)
- `AMPERE_24` - RTX 3070/3080/3090
- `AMPERE_48` - A40, RTX A6000
- `ADA_24` - L4, RTX 4000 series

```bash
export VIBEVOICE_GPU_TYPE="AMPERE_24"
```

### Voice Samples (VibeVoice)

For best results with voice cloning:
- Use clean audio samples (3-10 seconds)
- WAV format recommended
- Clear speech without background noise
- Sample rate of 16kHz or higher

## Architecture

### Project Structure

```
article-tts/
├── src/article_tts/
│   ├── __init__.py           # Package initialization
│   ├── cli.py                # CLI interface
│   ├── utils.py              # Shared utilities
│   ├── runpod_manager.py     # Runpod API management
│   └── models/
│       ├── __init__.py       # Model exports
│       ├── chatterbox.py     # ChatterboxTurboTTS client
│       └── vibevoice.py      # VibeVoice client
├── runpod_deployments/
│   └── vibevoice/
│       ├── handler.py        # Runpod serverless handler
│       ├── Dockerfile        # Container image
│       └── requirements.txt  # Handler dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

### How It Works

1. **ChatterboxTurboTTS**:
   - Uses your pre-configured Runpod endpoint
   - Splits text into 100-token chunks
   - Sends chunks to Runpod for inference
   - Concatenates audio on client-side

2. **VibeVoice**:
   - First run: Creates template and endpoint via GraphQL API
   - Uses Docker image `tejaskale/vibevoice-runpod:latest`
   - VibeVoice package installed in container automatically
   - Handles text chunking (500 chars) on server-side
   - Returns complete audio via API

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

### Building VibeVoice Docker Image

If you want to build your own VibeVoice image:

```bash
cd runpod_deployments/vibevoice
docker build -t your-username/vibevoice-runpod:latest .
docker push your-username/vibevoice-runpod:latest
```

Then use it:

```bash
export VIBEVOICE_DOCKER_IMAGE="your-username/vibevoice-runpod:latest"
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

### General Issues

**"Runpod API key required"**
- Set `RUNPOD_API_KEY` environment variable
- Or pass `--runpod-api-key` option

### ChatterboxTurboTTS Issues

**"Runpod credentials required"**
- Ensure both `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` are set
- Verify your endpoint exists in Runpod console

**"Job timed out"**
- Check Runpod endpoint status
- Ensure endpoint has workers available
- Try shorter text

### VibeVoice Issues

**"Template creation failed"**
- Check your Runpod API key has permissions
- Verify network connectivity
- Check Runpod status page

**"Endpoint not starting"**
- May take 2-3 minutes on first run (cold start)
- Check Runpod console for endpoint status
- Try a different GPU type if unavailable

**"Audio quality issues with voice samples"**
- Use higher quality voice samples (16kHz+)
- Ensure sample is clean (no background noise)
- Try samples between 3-10 seconds
- WAV format recommended

**"Job timed out after 10 minutes"**
- Text may be too long
- Try shorter text or manual chunking
- Check Runpod endpoint logs

## Cost Estimation

Runpod serverless pricing varies by GPU:

- **AMPERE_16** (RTX 3060): ~$0.14/hour
- **AMPERE_24** (RTX 3080): ~$0.20/hour
- **AMPERE_48** (A40): ~$0.35/hour

Serverless workers scale to zero when idle, so you only pay for active inference time.

## License

MIT License - See LICENSE file for details

## Credits

- **ChatterboxTurboTTS**: Based on the Chatterbox model
- **VibeVoice**: Based on the VibeVoice project
- **Runpod**: Serverless GPU infrastructure

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
- [Runpod Platform](https://runpod.io)

## References

Based on research from:
- [Runpod Documentation - Manage Templates](https://docs.runpod.io/sdks/graphql/manage-pod-templates)
- [Runpod Documentation - Manage Endpoints](https://docs.runpod.io/sdks/graphql/manage-endpoints)
- [RunPod Python SDK](https://github.com/runpod/runpod-python)
- [RunPod GraphQL API Spec](https://graphql-spec.runpod.io/)
