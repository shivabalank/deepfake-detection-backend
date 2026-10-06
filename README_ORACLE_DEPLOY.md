# Oracle Cloud Deployment Guide

Migrates the backend from Google Cloud Run to a persistent Oracle Cloud
Always Free VM. The Flask app, API endpoints, model, and inference logic
are **unchanged** — this only changes how/where the container runs.

```
Vercel Frontend  -->  Oracle Cloud VM  -->  Docker / Gunicorn / Flask  -->  Hugging Face (model)
```

---

## 1. Files changed

| File | Change |
|---|---|
| `requirements.txt` | Removed `--extra-index-url https://download.pytorch.org/whl/cpu` — that index has no ARM (aarch64) wheels and would break install on Oracle's free Ampere A1 shape. Standard PyPI `torch`/`torchvision` wheels are used instead (no functional change to the model). |
| `Dockerfile` | Comments/defaults updated for a persistent VM instead of Cloud Run's serverless model. Still uses gunicorn, still listens on `$PORT` (default 8080). No change to how the app itself runs. |
| `app.py` | Added an **optional** `ALLOWED_ORIGINS` env var for CORS. If unset, behavior is identical to before (`CORS(app)`, any origin allowed) — nothing changes unless you explicitly set it. |
| `.dockerignore` | **New.** Keeps the build context lean (excludes venv, dataset, checkpoints, etc.). |
| `docker-compose.yml` | **New.** Runs the container with `restart: unless-stopped` — this is what fixes "backend always disconnects," since the VM stays on and Docker auto-restarts the service if it ever crashes or the VM reboots. |
| `.env.example` | **New.** Documents `MODEL_URL`, `PORT`, and optional `ALLOWED_ORIGINS` — no real secrets committed. |

Nothing was changed in: model/inference code, `/health`, `/predict/image`,
`/predict/video`, `/report/<name>`, `/output/<path>`, the frontend, or the
Hugging Face model hosting setup.

---

## 2. Exact content of new/changed files

All four new files (`.dockerignore`, `docker-compose.yml`, `.env.example`,
and this guide) plus the two edited files (`requirements.txt`, `Dockerfile`,
`app.py`) are included in the project zip provided alongside this guide —
copy them directly into your repo.

---

## 3. Oracle Cloud VM setup steps

### 3.1 Create the VM

1. Sign up / log in at [cloud.oracle.com](https://cloud.oracle.com)
2. Go to **Compute → Instances → Create Instance**
3. **Image**: Canonical **Ubuntu 22.04** (or latest LTS)
4. **Shape** — click **Change Shape**, choose:
   - **Ampere (ARM-based)** → **VM.Standard.A1.Flex**
   - Set **2 OCPUs** and **12 GB RAM** (Oracle's Always Free tier includes up to 4 OCPUs / 24GB total across A1 instances — 2/12 leaves room for a second free instance later if you want)
   - **Why ARM, not the AMD micro shape**: the free AMD shape (`VM.Standard.E2.1.Micro`) only has **1GB RAM** — likely too tight for PyTorch + EfficientNet-B4 + MTCNN, the same class of problem you hit on Cloud Run's 512MB/2GB limits. The ARM A1 shape gives you dramatically more headroom, for free.
5. **Add SSH key**: generate or upload one (Oracle's console can generate a key pair for you — download and save the private key)
6. **Networking**: leave default VCN/subnet, ensure **"Assign a public IPv4 address"** is checked
7. Click **Create**

### 3.2 Connect to the VM

```bash
ssh -i /path/to/your-private-key.pem ubuntu@<VM_PUBLIC_IP>
```

### 3.3 Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```
Log out and back in (or run `newgrp docker`) so your user can run `docker` without `sudo`.

### 3.4 Clone your backend repo

```bash
git clone https://github.com/shivabalank/deepfake-detection-backend.git
cd deepfake-detection-backend
```

### 3.5 Create your `.env` file

```bash
cp .env.example .env
nano .env
```
Set `MODEL_URL` to your actual Hugging Face **resolve** link (not `/blob/`):
```
MODEL_URL=https://huggingface.co/ShivaBalan-k/deepfake-detector-b4/resolve/main/best_model.pth
```
Save and exit (`Ctrl+O`, Enter, `Ctrl+X` in nano).

---

## 4. Docker build/run commands

Using docker-compose (recommended — handles the restart policy):
```bash
docker compose up -d --build
```

To check it's running:
```bash
docker compose ps
docker compose logs -f
```

To stop/restart:
```bash
docker compose down
docker compose up -d
```

(Equivalent manual `docker build`/`docker run` commands, if you prefer not to use compose:)
```bash
docker build -t deepfake-detection-backend .
docker run -d --name deepfake-detection-backend \
  --restart unless-stopped \
  -p 8080:8080 \
  --env-file .env \
  deepfake-detection-backend
```

---

## 5. Firewall / security ports to open

**Two separate firewall layers exist on Oracle Cloud — both must allow port 8080, this is the most common setup mistake:**

### 5.1 Oracle Cloud's network-level firewall (Security List)

1. Console → **Networking → Virtual Cloud Networks** → your VCN → your subnet → **Security Lists** → default security list
2. **Add Ingress Rule**:
   - Source CIDR: `0.0.0.0/0`
   - IP Protocol: TCP
   - Destination Port Range: `8080`
3. Save

### 5.2 The VM's own OS-level firewall (iptables)

Oracle's Ubuntu images ship with restrictive `iptables` rules by default. SSH into the VM and run:
```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8080 -j ACCEPT
sudo netfilter-persistent save
```
(If `netfilter-persistent` isn't installed: `sudo apt-get install -y iptables-persistent` first, then re-run the save command.)

If you skip either layer, you'll be able to reach the VM via SSH but `/health` will time out from outside — that mismatch is the telltale sign one of these two steps was missed.

---

## 6. Getting your backend's public URL

Your backend is reachable directly at:
```
http://<VM_PUBLIC_IP>:8080
```
Find `<VM_PUBLIC_IP>` on the VM's instance details page in the Oracle Cloud console.

(Optional, for a proper HTTPS domain instead of a raw IP: point a domain's DNS A record at this IP, then set up a reverse proxy with Caddy or Nginx + Let's Encrypt in front of the container. Not required to get things working — your Vercel frontend can call the plain `http://` IP directly — but worth doing later for a cleaner public-facing URL and to avoid mixed-content browser warnings if your frontend itself needs HTTPS-only requests.)

---

## 7. Vercel environment variable change

In your Vercel project → **Settings → Environment Variables**:
- Update `VITE_API_BASE_URL` to:
  ```
  http://<VM_PUBLIC_IP>:8080
  ```
- **Deployments** tab → latest deployment → **⋯** → **Redeploy** (environment variable changes require a rebuild to take effect)

---

## 8. Test `/health`

```bash
curl http://<VM_PUBLIC_IP>:8080/health
```
Expected:
```json
{"status": "ok", "device": "cpu"}
```
Or just visit that URL in a browser.

---

## 9. Test image prediction

```bash
curl -X POST http://<VM_PUBLIC_IP>:8080/predict/image \
  -F "file=@/path/to/a/test_image.jpg"
```
Expected: a JSON response with `prediction`, `confidence`, `conclusion`, `heatmap_image`, and `report_data_uri` fields — same shape as before, unchanged.

Also test end-to-end from your live Vercel site once the env var is updated and redeployed.

---

## 10. Verifying the Hugging Face model downloads correctly

Check the container logs right after first startup:
```bash
docker compose logs backend | grep -i checkpoint
```
You should see something like:
```
Checkpoint not found locally — downloading from https://huggingface.co/.../resolve/main/best_model.pth ...
Downloaded checkpoint to ./checkpoints/best_model.pth (XX.X MB)
Loaded trained weights from ./checkpoints/best_model.pth
```
If the downloaded size looks suspiciously small (under 1MB), the `MODEL_URL` likely points at a `/blob/` page instead of `/resolve/` — double check `.env`.

---

## 11. ARM/AMD compatibility notes (PyTorch, torchvision, facenet-pytorch, OpenCV)

- **torch / torchvision**: official PyPI wheels support Linux **aarch64** (ARM64) for CPU inference — `pip install torch torchvision` works directly on the Ampere A1 shape with no special index needed (this is exactly why the x86-only `--extra-index-url` line was removed from `requirements.txt`).
- **facenet-pytorch** (MTCNN): pure Python on top of torch — no architecture-specific binaries, works the same on ARM as x86.
- **opencv-python-headless**: publishes aarch64 wheels on PyPI — installs natively, no compilation needed.
- **Pillow, numpy, scikit-learn, reportlab, Flask, gunicorn**: all have standard aarch64 wheels available.
- **If you ever switch to the free AMD shape instead** (`VM.Standard.E2.1.Micro`, 1GB RAM — not recommended given memory constraints): that shape is x86_64, so the removed `--extra-index-url` line could optionally be restored there to get smaller CPU-only torch wheels instead of the default CUDA-bundled ones — but given the 1GB RAM ceiling, this shape likely isn't viable for this model regardless of wheel size.

---

## 12. End-to-end testing checklist

- [ ] VM created with Ampere A1 shape, 2+ OCPUs, 12+ GB RAM
- [ ] Docker installed and running (`docker --version`)
- [ ] Repo cloned, `.env` created with correct `MODEL_URL` (using `/resolve/`, not `/blob/`)
- [ ] `docker compose up -d --build` completes without errors
- [ ] Oracle Security List ingress rule added for port 8080
- [ ] VM's `iptables` rule added and saved for port 8080
- [ ] `curl http://<VM_PUBLIC_IP>:8080/health` returns `{"status": "ok", ...}` from an **external** machine (not just from inside the VM via SSH)
- [ ] Container logs confirm the model downloaded at a realistic file size (tens of MB, not KB)
- [ ] `docker compose logs -f` shows no repeated crash/restart loop
- [ ] Vercel `VITE_API_BASE_URL` updated to the new VM URL, redeployed
- [ ] Live Vercel site: image upload → prediction → Grad-CAM heatmap → Download Report all work end-to-end
- [ ] VM reboot test: `sudo reboot`, wait ~1 min, re-test `/health` — confirms `restart: unless-stopped` brings the container back automatically
