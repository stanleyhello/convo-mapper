# Convo-Mapper Quick Start Guide

## For Hackathon Teammates

This guide helps you quickly set up and run convo-mapper for the hackathon.

## Prerequisites

- **Docker Desktop** installed and running
- **Git** installed
- **GitHub account** (for pulling Docker images)

---

## Option 1: Using Docker (Recommended)

### Standard Setup (Single Computer)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/stanleyhello/convo-mapper.git
   cd convo-mapper
   ```

2. **Configure environment (optional):**
   ```bash
   cp .env.example .env
   # Edit .env to customize settings (mic name, model size, etc.)
   ```

3. **Start the application:**
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

4. **Access the web interface:**
   - Open http://localhost:5000 in your browser
   - You should see live transcription from your microphone and system audio

5. **View logs:**
   ```bash
   docker-compose logs -f
   ```

6. **Stop the application:**
   ```bash
   docker-compose down
   ```

---

### Hackathon Multi-Computer Setup

#### Control Computer (Friend's Computer)

Runs both Parallax scheduler and convo-mapper:

```bash
git clone https://github.com/stanleyhello/convo-mapper.git
cd convo-mapper

# Start Parallax + convo-mapper
docker-compose -f docker-compose.control.yml up -d

# Access web UI at http://localhost:5000
```

#### Worker Computers (Your Computer + Others)

Joins the Parallax cluster as a worker:

```bash
git clone https://github.com/stanleyhello/convo-mapper.git
cd convo-mapper

# Set the control computer's IP address
export SCHEDULER_ADDRESS=192.168.1.100  # Replace with actual IP

# Start Parallax worker
docker-compose -f docker-compose.worker.yml up -d
```

**To find the control computer's IP:**
```bash
# On macOS/Linux:
ifconfig | grep "inet " | grep -v 127.0.0.1

# On Windows:
ipconfig | findstr IPv4
```

---

## Option 2: Running Natively (macOS/Linux)

If Docker audio isn't working or you prefer native execution:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Parallax separately:**
   ```bash
   # In a separate terminal:
   parallax run -m Qwen/Qwen3-0.6GB -n 1
   ```

3. **Run the application:**
   ```bash
   # For web UI + transcription:
   python main.py

   # For web UI + transcription + topic summarization:
   python controller.py
   ```

4. **Access:** http://localhost:5000

---

## Configuration

Edit `.env` or set environment variables:

### Audio Settings
- `SAMPLE_RATE` - Audio sample rate (default: 16000)
- `MIC_NAME_FILTER` - Microphone name filter (default: "Yeti")
- `SPEAKER_NAME_FILTER` - Speaker name filter (optional)

### Whisper Model
- `MODEL_NAME` - Model size: tiny, base, small, medium, large-v3 (default: "small")
- `LANGUAGE` - Language code (default: "en")
- `WHISPER_DEVICE_PREFERENCE` - "cpu" or "cuda" (default: "cpu")

### Topic Summarization
- `PARALLAX_ENDPOINT` - Parallax LLM endpoint (default: http://localhost:3001/v1)
- `SUMMARY_INTERVAL_SECONDS` - Topic summary interval (default: 60)
- `TOPIC_MODEL` - Model for topic extraction (default: Qwen/Qwen3-0.6GB)

---

## Troubleshooting

### Docker not starting
```bash
# Check if Docker is running:
docker info

# If not running, start Docker Desktop
```

### Audio not working in Docker (Linux)
```bash
# Check PulseAudio is running:
pulseaudio --check
pulseaudio --start

# List audio devices:
docker exec convo-mapper python -c "import soundcard; print(soundcard.all_microphones())"
```

### Can't connect to Parallax
```bash
# Check Parallax is running:
curl http://localhost:3001/v1/models

# Verify network mode in docker-compose.yml is set to "host"
```

### macOS audio issues
Docker on macOS doesn't support direct audio device access. **Use native installation** instead:
```bash
pip install -r requirements.txt
python controller.py
```

### Wrong microphone selected
Edit `.env` and set `MIC_NAME_FILTER` to match your mic:
```bash
# Find your microphone name:
python -c "import soundcard as sc; [print(m.name) for m in sc.all_microphones()]"

# Then update .env:
MIC_NAME_FILTER=YourMicName
```

---

## Quick Commands Reference

```bash
# Pull latest image
docker-compose pull

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Stop services
docker-compose down

# Check status
docker-compose ps

# Update to latest version
git pull origin main
docker-compose pull
docker-compose up -d
```

---

## Support

- **Check logs:** `docker-compose logs -f convo-mapper`
- **GitHub Issues:** https://github.com/stanleyhello/convo-mapper/issues
- **CI/CD Plan:** See `CI/CD_Plan.md` for detailed architecture

---

## Privacy Note

🔒 **All processing happens locally.** Your audio never leaves your computer.
- Transcription: Local faster-whisper
- LLM: Local Parallax instance
- Storage: Local only (in-memory by default)
