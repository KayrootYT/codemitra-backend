# Oracle Cloud deployment (zero-budget target)

This deploys the existing FastAPI service with Docker Compose and Caddy. It is designed for an Oracle Cloud Always Free eligible Ubuntu VM. Free-tier eligibility, regional capacity, resource limits and availability are controlled by Oracle and are not guaranteed.

## Before starting

- Create an Ubuntu ARM64 VM using only resources explicitly marked **Always Free eligible** in your Oracle account. The A1 shape has historically offered up to 4 OCPUs / 24 GB RAM in the free allowance, subject to tenancy and regional availability; do not assume capacity is available.
- Open inbound TCP 22 (SSH), TCP 80 and TCP 443 in the VM's network security list/security list and Ubuntu firewall. Do not expose port 8000 publicly.
- Point a DNS A record such as `api.example.com` to the VM's public IPv4 address. Caddy obtains and renews a public TLS certificate automatically. A stable custom hostname/domain is needed for this production HTTPS setup; domain registration may cost money.
- Install Docker Engine and the Docker Compose plugin using Docker's official Ubuntu instructions.

## Deploy

From the repository root on the VM:

```bash
git clone https://github.com/KayrootYT/codemitra-backend.git
cd codemitra-backend
cp .env.example .env
nano .env
```

Set at minimum:

- `NVIDIA_API_KEY`: your inference provider key (required for AI chat; hosting can be free while inference may have its own limits/costs).
- `API_DOMAIN`: the DNS hostname pointing to this VM.
- `ALLOWED_ORIGINS=https://kayrootyt.github.io`
- `TTS_PROVIDER=piper`

Then start:

```bash
docker compose -f deploy/oracle-cloud/compose.yaml up -d --build
docker compose -f deploy/oracle-cloud/compose.yaml ps
docker compose -f deploy/oracle-cloud/compose.yaml logs -f api
```

Test:

```bash
curl https://YOUR_API_DOMAIN/health
curl https://YOUR_API_DOMAIN/voice/config
```

The first Hindi speech request downloads the selected Piper model and its JSON config into the persistent Docker volume. The download is not repeated after container restarts unless the volume is deleted. Male and female presets are `hi_IN-rohan-medium` and `hi_IN-priyamvada-medium`.

## Frontend connection

Open CodeMitra in GitHub Pages, go to Settings, set **Backend server URL** to `https://YOUR_API_DOMAIN`, save, then run the health check. The app stores this setting in the browser. Keep the GitHub Pages origin in `ALLOWED_ORIGINS`.

## GPU voice is optional

This deployment intentionally keeps Piper as the reliable CPU Hindi voice. A free GPU endpoint is not included in this Compose stack: Hugging Face ZeroGPU has quotas/queues and is a separate Gradio service, not a general-purpose GPU server. Add it later behind a dedicated adapter with Piper fallback; never make the app depend on the GPU endpoint for basic speech.

## Operations

```bash
# Update from the selected branch
git pull
docker compose -f deploy/oracle-cloud/compose.yaml up -d --build

# Logs / status
docker compose -f deploy/oracle-cloud/compose.yaml logs --tail=200 api
docker compose -f deploy/oracle-cloud/compose.yaml ps

# Stop (keeps persistent Piper and Caddy volumes)
docker compose -f deploy/oracle-cloud/compose.yaml down
```

Back up the VM configuration and keep `.env` private. Do not commit secrets. Configure Oracle budget alerts/quotas where available and verify every resource is Always Free before creating it.

## ARM64 compatibility check

Build and start on the actual ARM64 VM before treating deployment as complete. Python wheels for ML/audio dependencies can vary by architecture. If `piper-tts` or one of its ONNX dependencies has no compatible ARM64 wheel, do not silently switch to a paid instance: keep chat on the VM and run Piper on a compatible CPU host or use the browser speech fallback until an ARM-compatible build is available.
