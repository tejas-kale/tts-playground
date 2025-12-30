# VibeVoice Runpod Deployment

This directory contains the Runpod serverless deployment for VibeVoice TTS.

## Files

- `handler.py` - Runpod serverless handler for VibeVoice inference
- `Dockerfile` - Container image for deployment
- `requirements.txt` - Python dependencies

## Automatic Deployment

The `article-tts` CLI automatically handles template creation and endpoint deployment when you first use VibeVoice. You don't need to manually deploy this.

## Manual Deployment

If you want to deploy manually:

1. Build and push the Docker image:
```bash
docker build -t your-username/vibevoice-runpod:latest .
docker push your-username/vibevoice-runpod:latest
```

2. Create a template in Runpod console or use the GraphQL API

3. Deploy an endpoint using the template

## Handler Input Format

```json
{
  "input": {
    "text": "Text to synthesize",
    "voice_sample": "base64_encoded_audio_optional",
    "add_speaker_labels": true,
    "chunk_size": 500
  }
}
```

## Handler Output Format

```json
{
  "audio": "base64_encoded_wav_audio"
}
```

Or on error:
```json
{
  "error": "error_message"
}
```
