# CodeMitra — no-card Render deployment

This is the recommended no-credit-card setup for the current CodeMitra FastAPI backend.

- Frontend: GitHub Pages (free static hosting).
- Backend: Render Free Web Service using this repository's Dockerfile.
- Hindi voices: local CPU Piper voices (Rohan male, Priyamvada female).
- AI chat: requires an inference-provider API key. Free hosting does not make model inference automatically free.
- No custom domain is required; Render supplies an HTTPS `onrender.com` URL.

## Create the backend (no card)

1. Sign up at https://render.com using your GitHub account. Render's official first-deploy guide says no payment is required for the free deployment flow.
2. Open https://dashboard.render.com/blueprints and choose **New Blueprint Instance** (or **New + → Blueprint**).
3. Connect the `KayrootYT/codemitra-backend` GitHub repository.
4. Select branch `feature/codemitra-quality-upgrade` while this PR is still open.
5. Render detects `render.yaml`. Review the service and select the **Free** plan.
6. Add `NVIDIA_API_KEY` as a secret environment variable in the Render dashboard. Do not put it in GitHub or the frontend.
7. Apply the Blueprint and wait for the Docker build and deploy to finish.

If Blueprint creation is unavailable in your dashboard, create **New + → Web Service**, connect the same repository and branch, select **Docker** and **Free**, set the health check path to `/health`, and add these environment variables:

| Key | Value |
|---|---|
| `ALLOWED_ORIGINS` | `https://kayrootyt.github.io` |
| `TTS_PROVIDER` | `piper` |
| `PIPER_MODEL_DIR` | `/tmp/codemitra-piper` |
| `MODEL` | `deepseek-ai/deepseek-v4.1-flash` |
| `NVIDIA_API_KEY` | Your inference provider key (secret) |

The Docker command listens on Render's `PORT` environment variable. Do not hardcode port 8000 for Render.

## Connect the frontend

1. Copy the deployed service URL, for example `https://codemitra-api.onrender.com`.
2. Open CodeMitra → Settings.
3. Paste the URL into **Backend server URL**, save, and run the health check.
4. The frontend origin must remain in `ALLOWED_ORIGINS`: `https://kayrootyt.github.io`.

## Verify

- Open `https://YOUR-SERVICE.onrender.com/health`; expect `{"status":"healthy"}`.
- Open `/voice/config`; expect Piper and male/female voices to be available.
- Test Hindi chat, English chat, image/file/ZIP attachments, project ZIP export, and Hindi male/female speech.

## Free-plan limits you should know

- Render Free Web Services sleep after 15 minutes without inbound traffic. The first request after sleep can take about a minute while the service starts.
- Free instances share a monthly 750-hour workspace allowance. If the allowance or included usage is exhausted and no payment method is attached, Render suspends free services or new builds rather than charging a card.
- The filesystem is ephemeral. Piper voice files may need to download again after a restart, sleep, or redeploy. Do not use local disk for user uploads or permanent data.
- Render may restart free services. This is a hobby/testing deployment, not guaranteed 24×7 production hosting.

## Security

Keep `NVIDIA_API_KEY`, ElevenLabs credentials, and any other provider secrets only in Render's environment settings. Never paste secrets into chat, commit them to GitHub, or place them in frontend JavaScript.

Official references:
- https://render.com/docs/your-first-deploy
- https://render.com/docs/free
- https://render.com/docs/faq
