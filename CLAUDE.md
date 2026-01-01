# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Package Management**: This project uses `uv` as the package manager.

```bash
# Install the package in development mode
uv pip install -e ".[dev]"

# Run the CLI tool locally (after install)
tts-pg --help

# Code formatting and linting
black src/
ruff check src/

# Run a single file manually
python -m tts_pg.cli
```

**Testing**: Currently no test suite exists. When adding tests, use `pytest`:
```bash
pytest
pytest tests/test_specific_file.py
```

## Architecture Overview

### Core Design Pattern

This is a **serverless TTS CLI tool** that orchestrates remote GPU inference on Runpod. The architecture separates:
- **Local CLI** (src/tts_pg/): User interface and orchestration
- **Remote Handler** (runpod_deployments/): Serverless inference on GPU

### Key Architectural Concepts

#### 1. Dynamic Handler Deployment

The handler is **NOT** packaged with the CLI. Instead, it's downloaded at container startup:

```python
# In cli.py configure command
docker_args = (
    "/bin/bash -c '"
    "apt-get update && "
    "apt-get install -y ffmpeg && "
    "pip install runpod chatterbox-tts torchaudio accelerate && "
    "wget -O /handler.py https://raw.githubusercontent.com/.../handler.py && "
    "python /handler.py"
    "'"
)
```

**Why**: Enables independent handler updates without CLI reinstallation. The handler URL must point to the `main` branch, not feature branches.

**Critical**: The entire docker_args must be wrapped in `/bin/bash -c '...'` to prevent shell argument parsing errors (especially with wget's `-O` flag).

#### 2. Runpod GraphQL API Integration

All Runpod operations use the GraphQL API (not the Python SDK):
- Template creation: `saveTemplate` mutation
- Endpoint creation: `saveEndpoint` mutation
- Status monitoring: `myself.endpoints` query

See `runpod_manager.py` for the `RunpodManager` class that wraps all GraphQL operations.

#### 3. ChatterboxTurboTTS Token Chunking

ChatterboxTurboTTS has a **100 token limit**. Longer text requires:
1. Token counting (client-side approximation)
2. Sentence-aware splitting at boundaries (periods, commas)
3. Chunk synthesis (parallel requests)
4. Audio concatenation (torch.cat)

This logic exists in **both** the client (`chatterbox.py`) and handler (`handler.py`) to optimize network usage.

#### 4. Configuration Management

User configuration is stored at `~/.tts-pg/config.json`:
```json
{
  "api_key": "runpod-key",
  "endpoint_id": "xyz123",
  "template_id": "abc456",
  "docker_image": "runpod/pytorch:...",
  "gpu_type": "AMPERE_16"
}
```

The `Config` class (config.py) handles:
- JSON persistence
- Environment variable fallbacks (RUNPOD_API_KEY, HF_TOKEN)
- Masking sensitive values in display

#### 5. CLI Structure

Built with Click and Rich:
- `configure`: One-time Runpod setup (creates template + endpoint)
- `speak`: Text-to-speech synthesis (with chunking, speed adjustment, MP3 conversion)
- `preprocess`: Preview text processing
- `info`: Display model and config information

All commands use Rich console for colored output and progress indication.

### Component Responsibilities

**src/tts_pg/cli.py**: CLI commands and user interaction
**src/tts_pg/config.py**: Configuration persistence (~/.tts-pg/config.json)
**src/tts_pg/runpod_manager.py**: GraphQL API wrapper for Runpod operations
**src/tts_pg/models/chatterbox.py**: Client for ChatterboxTurboTTS via Runpod
**src/tts_pg/utils.py**: Audio processing (speed, MP3 conversion, playback)
**runpod_deployments/unified/handler.py**: Serverless inference handler (downloaded by container)

## Important Constraints and Patterns

### Handler Changes

When modifying `runpod_deployments/unified/handler.py`:
1. Changes only take effect after pushing to `main` branch
2. Existing endpoints continue using old handler until container restart
3. Test handler changes by creating a new endpoint or forcing worker restart

### First Request Delay

Cold start takes 3-5 minutes because the container must:
1. Install system dependencies (ffmpeg)
2. Install Python packages (chatterbox-tts, torchaudio, accelerate)
3. Download handler from GitHub
4. Load the ChatterboxTurboTTS model (1-2 GB)

Subsequent requests are fast because models stay cached in GPU memory.

### Token Chunking Algorithm

The chunking algorithm in both client and handler:
1. Splits at sentence boundaries (`(?<=[.!?])\s+`)
2. If sentence exceeds 100 tokens, splits at commas (`(?<=,)\s+`)
3. Tracks token count to stay under limit
4. Concatenates chunks with `torch.cat(tensors, dim=-1)`

**Never** modify chunking logic in only one location—client and handler must stay synchronized.

### Environment Variables

Required for endpoint configuration:
- `RUNPOD_API_KEY`: Runpod API authentication
- `HF_TOKEN`: Hugging Face token (required for ChatterboxTurboTTS model access)

These can be set in environment or passed via CLI flags.

## Project-Specific Notes

- **Python version**: Requires Python 3.10+ (see pyproject.toml)
- **GPU types**: AMPERE_16 (RTX 3060) is recommended default; supports AMPERE_24, AMPERE_48, ADA_24
- **Audio format**: Handler produces WAV at model sample rate; CLI can convert to MP3
- **Speed adjustment**: Handled client-side using pydub time stretching
- **Serverless scaling**: Workers scale to zero after 5 seconds idle (configurable)

## Common Pitfalls

1. **Handler URL must point to `main` branch** - Feature branches cause 404 errors when containers start
2. **Docker args must be wrapped in `/bin/bash -c`** - Otherwise shell parsing breaks on flags like `-O`
3. **Token chunking must match** - Client and handler chunking logic must be identical
4. **Don't skip dependency installation** - Handler must install packages every cold start since it uses base PyTorch image
