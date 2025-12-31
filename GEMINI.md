# TTS Playground

A unified CLI tool for high-quality text-to-speech synthesis using ChatterboxTurboTTS and VibeVoice, deployed via a single RunPod serverless endpoint.

## Project Overview

**Purpose:** Provide a seamless CLI experience for generating speech from text using advanced AI models, leveraging serverless GPU infrastructure for performance and scalability.

**Architecture:**
*   **Client (CLI):** Python-based CLI (`tts-pg`) that manages configuration, preprocessing, and API requests to RunPod.
*   **Server (RunPod):** A unified Docker container hosting both TTS models. The `handler.py` script routes requests to the appropriate model (`chatterbox` or `vibevoice`) within the same endpoint.

**Key Technologies:**
*   **Language:** Python 3.10+
*   **Infrastructure:** RunPod Serverless (GPU)
*   **Models:**
    *   **ChatterboxTurboTTS:** Fast, efficient text-to-speech.
    *   **VibeVoice:** High-quality voice cloning.
*   **CLI Framework:** `click`, `rich`
*   **Audio Processing:** `pydub`, `soundfile`
*   **Build System:** `hatchling`

## Directory Structure

```
tts-playground/
├── src/tts_pg/            # Main Python package
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point and commands
│   ├── config.py               # Configuration management (~/.tts-pg/config.json)
│   ├── runpod_manager.py       # RunPod API integration (template/endpoint management)
│   ├── utils.py                # Audio processing and file utilities
│   └── models/                 # Client-side model wrappers
├── runpod_deployments/         # Server-side deployment code
│   └── unified/
│       ├── handler.py          # Unified serverless handler for both models
│       ├── Dockerfile          # Container definition
│       └── requirements.txt    # Python dependencies for the handler
├── pyproject.toml              # Project metadata and dependencies
└── README.md                   # User documentation
```

## Setup and Usage

### Installation

1.  **From Source:**
    ```bash
    git clone https://github.com/tejas-kale/tts-playground.git
    cd tts-playground
    uv tool install .
    ```

### Configuration

Run the one-time configuration setup to deploy the RunPod endpoint:

```bash
tts-pg configure
```
*   Prompts for RunPod API Key.
*   Prompts for Hugging Face Token (required for Chatterbox).
*   Creates/Configures the RunPod Serverless Endpoint.

### Generating Speech

```bash
# Basic usage
tts-pg speak "Hello, world!" -m chatterbox

# Voice cloning (VibeVoice)
tts-pg speak -f text.txt -m vibevoice --voice-sample my_voice.wav

# Output formats
tts-pg speak "..." --format mp3 --bitrate 320k
```

## Development

### Environment Setup

1.  **Clone and Install:**
    ```bash
    git clone https://github.com/tejas-kale/tts-playground.git
    cd tts-playground
    uv pip install -e ".[dev]"
    ```

2.  **Linting & Formatting:**
    ```bash
    black src/
    ruff check src/
    ```

3.  **Testing:**
    ```bash
    pytest
    ```

### Server-Side Development (`runpod_deployments/unified/`)

*   **Handler Logic (`handler.py`):**
    *   Global model loading for cold start optimization.
    *   `handler(job)`: Entry point, routes based on `job["input"]["model"]`.
    *   `synthesize_chatterbox()`: Token-based chunking (100 tokens max).
    *   `synthesize_vibevoice()`: Sentence-based chunking (500 chars max), supports voice cloning.

*   **Building Custom Images:**
    If modifying `handler.py` or dependencies, you may need to rebuild and push the Docker image:
    ```bash
    cd runpod_deployments/unified
    docker build -t <your-repo>/tts-pg-unified:latest .
    docker push <your-repo>/tts-pg-unified:latest
    ```
    Then update configuration: `tts-pg configure --docker-image <your-repo>/tts-pg-unified:latest`

## Configuration Management (`src/tts_pg/config.py`)

*   Stores config in `~/.tts-pg/config.json`.
*   Key fields: `api_key`, `endpoint_id`, `template_id`, `docker_image`, `gpu_type`.
*   Prioritizes environment variables (e.g., `RUNPOD_API_KEY`) over config file values where applicable.
