# TTS Playground

A simple CLI tool for high-quality text-to-speech synthesis using ChatterboxTurboTTS via Runpod serverless infrastructure.

## Features

- **Serverless Deployment**: Runs on Runpod's serverless GPU infrastructure
- **One-Time Setup**: Configure once with `tts-pg configure`
- **Zero Local Dependencies**: No GPU or heavy ML libraries required locally
- **Fast Generation**: 10x real-time speed with ChatterboxTurboTTS
- **Intelligent Text Chunking**: Automatically splits long text at sentence boundaries
- **Speed Control**: Adjust playback speed (0.5x - 2.0x)
- **Format Options**: Output as WAV or MP3 with configurable bitrate
- **Auto-play**: Optional automatic audio playback after generation
- **Clean Interface**: Simple, intuitive CLI powered by Click

## Installation

Install using `uv`:

```bash
uv tool install tts-pg
```

Or install from source:

```bash
git clone https://github.com/tejas-kale/tts-playground.git
cd tts-playground
uv tool install .
```

## Quick Start

### Step 1: Configure

Run the configuration command to set up your Runpod endpoint:

```bash
tts-pg configure
```

You'll be prompted for:
- **Runpod API Key** (required) - Get from [runpod.io](https://runpod.io) Settings → API Keys
- **Hugging Face Token** (required) - Required for ChatterboxTurboTTS model access

This creates a serverless endpoint and saves configuration locally at `~/.tts-pg/config.json`.

### Step 2: Generate Speech

Once configured, you can start generating speech:

```bash
# Generate from text
tts-pg speak "Hello, world!"

# From a file
tts-pg speak -f article.txt

# Generate MP3 at custom speed
tts-pg speak -f book.txt --format mp3 --speed 1.0
```

## Usage

### Commands

#### `configure`

Set up the Runpod endpoint (run this once):

```bash
tts-pg configure [OPTIONS]
```

**Options:**
- `--api-key` - Runpod API key (or set `RUNPOD_API_KEY`)
- `--docker-image` - Base Docker image (default: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`)
- `--gpu-type` - GPU type: AMPERE_16, AMPERE_24, AMPERE_48, ADA_24 (default: AMPERE_16)
- `--workers-max` - Maximum workers (default: 1)
- `--hf-token` - Hugging Face token (or set `HF_TOKEN`)
- `--show-status` - Monitor endpoint deployment and startup status (blocking)

**Example:**
```bash
# Interactive (prompts for API key)
tts-pg configure

# With environment variables
export RUNPOD_API_KEY="your-key"
export HF_TOKEN="your-hf-token"
tts-pg configure

# With custom GPU
tts-pg configure --gpu-type AMPERE_24

# Monitor deployment status (blocking)
tts-pg configure --show-status
```

#### `speak`

Generate speech from text:

```bash
tts-pg speak [TEXT] [OPTIONS]
```

**Options:**
- `--file, -f` - Read text from file
- `--output, -o` - Output path (default: `/tmp/tts_pg_<timestamp>.wav`)
- `--format` - Output format: `wav` or `mp3` (default: wav)
- `--speed, -s` - Playback speed multiplier (default: 0.85)
- `--bitrate, -b` - MP3 bitrate: 128k, 192k, 320k (default: 192k)
- `--play/--no-play` - Auto-play audio (default: enabled)

**Examples:**
```bash
# Generate from text
tts-pg speak "Hello, world!"

# From file
tts-pg speak -f article.txt

# Generate MP3 at normal speed
tts-pg speak -f book.txt --format mp3 --speed 1.0

# Save without auto-play
tts-pg speak -f article.txt -o ~/audio/output.wav --no-play
```

#### `preprocess`

Preview how text will be processed:

```bash
tts-pg preprocess [TEXT] [OPTIONS]

# Examples
tts-pg preprocess "Hello, world!"
tts-pg preprocess -f article.txt
```

#### `info`

Display model information and configuration status:

```bash
tts-pg info
```

## ChatterboxTurboTTS

**Features:**
- 10x faster than standard TTS models
- Optimized 100-token chunking
- Automatic sentence-aware text splitting
- Seamless audio concatenation for long texts

**Performance:**
- Typical synthesis: ~0.1x real-time (10x faster)
- Cold start: 1-2 minutes (first request)
- Warm requests: < 5 seconds

## Configuration

### Configuration File

Configuration is stored at `~/.tts-pg/config.json`:

```json
{
  "api_key": "your-runpod-api-key",
  "endpoint_id": "xyz123",
  "template_id": "abc456",
  "docker_image": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
  "gpu_type": "AMPERE_16"
}
```

### Environment Variables

Alternatively, set these environment variables:

```bash
# Required for configuration
export RUNPOD_API_KEY="your-api-key"
export HF_TOKEN="your-hugging-face-token"
```

### GPU Types

Choose based on your performance/cost needs:

- `AMPERE_16` - RTX 3060, A2000, A4000 (~$0.14/hour) **Recommended**
- `AMPERE_24` - RTX 3070/3080/3090 (~$0.20/hour)
- `AMPERE_48` - A40, RTX A6000 (~$0.35/hour)
- `ADA_24` - L4, RTX 4000 series

Serverless workers scale to zero when idle - you only pay for active inference time.

## Architecture

### Serverless Deployment

ChatterboxTurboTTS runs on Runpod serverless infrastructure:

```
┌─────────────────────────────────────┐
│   Runpod Serverless Endpoint        │
│                                      │
│  ┌────────────────────────────────┐ │
│  │     ChatterboxTurboTTS         │ │
│  │    (100-token chunks)          │ │
│  │                                 │ │
│  │  • Automatic text splitting    │ │
│  │  • Sentence-aware chunking     │ │
│  │  • Audio concatenation         │ │
│  └────────────────────────────────┘ │
│                                      │
│  Handler processes requests          │
└─────────────────────────────────────┘
           ▲
           │ HTTPS API
           │
    ┌──────┴───────┐
    │   tts-pg     │
    │     CLI      │
    └──────────────┘
```

### Project Structure

```
tts-playground/
├── src/tts_pg/
│   ├── __init__.py
│   ├── cli.py               # CLI commands
│   ├── config.py            # Configuration management
│   ├── runpod_manager.py    # Runpod API integration
│   ├── utils.py             # Shared utilities
│   └── models/
│       ├── __init__.py
│       └── chatterbox.py    # ChatterboxTurboTTS client
├── runpod_deployments/
│   └── unified/
│       ├── handler.py       # Serverless handler
│       └── README.md        # Deployment docs
├── pyproject.toml
└── README.md
```

### How It Works

1. **First Run**: `tts-pg configure`
   - Creates Runpod template via GraphQL API
   - Uses base PyTorch image with dynamic installation
   - Start command installs ChatterboxTTS and dependencies
   - Downloads handler from GitHub
   - Deploys endpoint and saves endpoint ID to `~/.tts-pg/config.json`

2. **Container Startup** (automatic on first request):
   - Installs system dependencies (ffmpeg)
   - Installs Python packages (chatterbox-tts, torchaudio, accelerate)
   - Downloads and runs handler
   - Loads ChatterboxTurboTTS model (1-2 minutes first time)

3. **Synthesis**: `tts-pg speak`
   - Reads endpoint ID from config
   - Sends request to Runpod endpoint
   - Handler chunks text and generates audio
   - Returns generated audio as base64
   - CLI saves to file and optionally plays

## Development

### Install for Development

```bash
git clone https://github.com/tejas-kale/tts-playground.git
cd tts-playground

# Install with dev dependencies
uv pip install -e ".[dev]"
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

## Troubleshooting

### Configuration Issues

**"TTS Playground is not configured"**
- Run `tts-pg configure` first
- Check that `~/.tts-pg/config.json` exists

**"Configuration is incomplete"**
- Delete `~/.tts-pg/config.json`
- Run `tts-pg configure` again

**"Template creation failed"**
- Verify your Runpod API key has permissions
- Check Runpod status page for outages
- Try a different GPU type

### Synthesis Issues

**"Job timed out after 10 minutes"**
- Text may be too long - try shorter segments
- Check Runpod endpoint status in console
- Endpoint may be cold-starting (first request takes longer)

**ChatterboxTurboTTS errors**
- Ensure HF_TOKEN was provided during configuration
- Check Hugging Face token has access to Chatterbox model
- Reconfigure with correct token

### General Issues

**"Endpoint not found"**
- Check endpoint exists in Runpod console
- Run `tts-pg configure` to create new endpoint
- Verify endpoint ID in `~/.tts-pg/config.json`

**Cold start delays (1-2 minutes)**
- Normal for first request or after idle timeout
- Subsequent requests are fast (model cached)
- Increase idle timeout or min workers to keep warm
- Use `tts-pg configure --show-status` to monitor deployment progress

## Output Location

Default: `/tmp/tts_pg_<timestamp>.wav`

Custom location:
```bash
tts-pg speak -f article.txt -o ~/audio/output.wav
```

## Cost Estimation

**Runpod Serverless Pricing** (approximate):

| GPU Type | Cost/Hour | Best For |
|----------|-----------|----------|
| AMPERE_16 | ~$0.14 | General use (recommended) |
| AMPERE_24 | ~$0.20 | Faster processing |
| AMPERE_48 | ~$0.35 | High concurrency |
| ADA_24 | Varies | Latest architecture |

**Serverless benefits:**
- Scale to zero when idle (no cost)
- Pay only for active inference time
- Typical synthesis: $0.001-0.01 per request

## License

MIT License - See LICENSE file for details

## Credits

- **ChatterboxTurboTTS**: High-speed TTS model
- **Runpod**: Serverless GPU infrastructure

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Links

- [GitHub Repository](https://github.com/tejas-kale/tts-playground)
- [Issue Tracker](https://github.com/tejas-kale/tts-playground/issues)
- [Runpod Platform](https://runpod.io)
- [Runpod Documentation](https://docs.runpod.io)

## References

Implementation based on:
- [Runpod GraphQL API - Manage Templates](https://docs.runpod.io/sdks/graphql/manage-pod-templates)
- [Runpod GraphQL API - Manage Endpoints](https://docs.runpod.io/sdks/graphql/manage-endpoints)
- [RunPod Python SDK](https://github.com/runpod/runpod-python)
