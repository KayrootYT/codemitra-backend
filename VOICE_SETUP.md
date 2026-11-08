# CodeMitra natural voice setup

CodeMitra keeps provider credentials on the backend. Never put an ElevenLabs key in the GitHub Pages frontend.

## Render environment variables

Keep the existing `NVIDIA_API_KEY` and add:

- `ELEVENLABS_API_KEY`: API key from your ElevenLabs account.
- `VOICE_FEMALE_ID`: voice ID for the female preset you choose in your ElevenLabs account.
- `VOICE_MALE_ID`: voice ID for the male preset you choose in your ElevenLabs account.

Redeploy the backend after saving environment variables. The app falls back to device speech synthesis if natural TTS is not configured.

## App features

- Natural speech uses ElevenLabs TTS through `POST /voice/speak`.
- Female and male presets use the configured voice IDs.
- Friendly, calm, energetic, and professional styles adjust delivery settings.
- The browser speech engine is used as a fallback.
- Voice cloning uses `POST /voice/clone`. The person whose voice is cloned must be the uploader or must have explicitly authorized the clone. The UI requires a consent confirmation before upload.
- Keep the original recording and permission evidence. Do not clone public figures or other people without permission.

## Recording guidance

Use a clean, single-speaker recording with minimal background noise. About 1–2 minutes is a good target for a consistent sample. The upload limit in this app is 20 MB. ElevenLabs may apply its own account, plan, and policy limits.

## Costs and privacy

ElevenLabs API usage may be billed separately under your provider plan. Audio samples are sent to ElevenLabs to create the requested voice profile; review its current privacy and retention terms before uploading sensitive recordings. The backend does not save the uploaded audio locally.
