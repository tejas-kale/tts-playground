# Unified Article TTS Runpod Deployment

This directory contains the unified Runpod serverless deployment that supports both ChatterboxTurboTTS and VibeVoice models in a single endpoint.

## Files

- `handler.py` - Unified serverless handler supporting both models
- `Dockerfile` - Container image with both Chatterbox and VibeVoice (optional)
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Default Deployment Method

By default, `article-tts configure` uses **dynamic installation** with the base PyTorch image:

- **Base Image**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- **Start Command**: Installs dependencies and downloads handler at runtime
- **No pre-built images required**
- **Handler fetched from**: GitHub repository on container startup

This approach:
- ✅ No need to build or maintain custom Docker images
- ✅ Always uses latest handler code
- ✅ Simpler deployment process
- ⚠️ First cold start takes 3-5 minutes (subsequent starts are fast with cached models)

## Alternative: Pre-built Docker Image

If you prefer faster cold starts, you can build a custom image:

### 1. Build and Push Docker Image

```bash
# Build
docker build -t your-username/article-tts-unified:latest .

# Push to Docker Hub
docker push your-username/article-tts-unified:latest
```

### 2. Create Template in Runpod

Via Runpod console or use the CLI:

```bash
article-tts configure --docker-image your-username/article-tts-unified:latest
```

### 3. Deploy Endpoint

The configure command also deploys the endpoint automatically.

## Handler Input Format

The handler accepts requests with the following format:

### ChatterboxTurboTTS Request

```json
{
  "input": {
    "model": "chatterbox",
    "text": "Text to synthesize"
  }
}
```

### VibeVoice Request

```json
{
  "input": {
    "model": "vibevoice",
    "text": "Text to synthesize",
    "voice_sample": "base64_encoded_audio_optional",
    "add_speaker_labels": true,
    "chunk_size": 500
  }
}
```

## Handler Output Format

Success:
```json
{
  "audio": "base64_encoded_wav_audio"
}
```

Error:
```json
{
  "error": "error_message"
}
```

## Model Loading

Both models are loaded on cold start and cached for subsequent requests:

- **ChatterboxTurboTTS**: Requires `HF_TOKEN` environment variable
- **VibeVoice**: Loads from Hugging Face by default

## Optimizations

### ChatterboxTurboTTS
- Automatic text chunking at 100 token boundaries
- Sentence-aware splitting
- Fallback to comma splitting for long sentences
- Audio concatenation on server-side

### VibeVoice
- Configurable chunk size (default: 500 characters)
- Automatic speaker label injection
- Voice sample support via base64 encoding
- Seamless audio concatenation with pydub

## Environment Variables

Set these when creating the template:

- `HF_TOKEN` - Hugging Face token (required for Chatterbox)
- `VIBEVOICE_MODEL` - VibeVoice model name (optional, default: "seshurajup/VibeVoice")

## GPU Requirements

Recommended GPU types:
- **AMPERE_16** - RTX 3060 (~$0.14/hour) - Good for general use
- **AMPERE_24** - RTX 3080 (~$0.20/hour) - Better for high throughput
- **AMPERE_48** - A40 (~$0.35/hour) - Best for heavy loads

Both models fit comfortably on AMPERE_16 GPUs.

## Testing Locally

You can test the handler locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN="your-token"

# Run handler locally
python handler.py --rp_serve_api
```

Then send test requests:

```bash
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "model": "chatterbox",
      "text": "Hello, world!"
    }
  }'
```

## Troubleshooting

### "Could not find tokenizer"
- Check that ChatterboxTurboTTS is properly installed
- Verify HF_TOKEN is set and valid

### "VibeVoice is not installed"
- Ensure the Dockerfile properly clones and installs VibeVoice
- Check git clone command succeeded in build logs

### Out of memory errors
- Use larger GPU type (AMPERE_24 or AMPERE_48)
- Reduce chunk sizes in requests
- Check both models aren't loaded simultaneously unnecessarily

### Cold start timeout
- First request may take 2-3 minutes to load models
- Subsequent requests are fast (models cached)
- Increase idle timeout to keep models warm
