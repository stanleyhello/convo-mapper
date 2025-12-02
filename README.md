# Convo-Mapper

Real-time audio transcription and topic analysis for conversations. Fully private and locally run.

[![CI](https://github.com/stanleyhello/convo-mapper/workflows/CI/badge.svg)](https://github.com/stanleyhello/convo-mapper/actions/workflows/ci.yml)
[![Pre-commit](https://github.com/stanleyhello/convo-mapper/workflows/Pre-commit%20Checks/badge.svg)](https://github.com/stanleyhello/convo-mapper/actions/workflows/pre-commit.yml)
[![Docker Build](https://github.com/stanleyhello/convo-mapper/workflows/Build%20and%20Push%20Docker%20Images/badge.svg)](https://github.com/stanleyhello/convo-mapper/actions/workflows/docker-build.yml)

## Features

- 🎤 **Real-time transcription** of microphone and system audio
- 🔒 **Fully private** - all processing happens locally
- 🤖 **AI-powered topic detection** via local Parallax LLM
- 🌐 **Live web interface** for viewing transcripts
- 🐳 **Docker-ready** for easy deployment across teams

## Quick Start

### Using Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/stanleyhello/convo-mapper.git
cd convo-mapper

# Quick start script
chmod +x scripts/quick-start.sh
./scripts/quick-start.sh

# Or manually:
docker-compose up -d
```

**Web UI:** http://localhost:5000

### Native Installation

```bash
pip install -r requirements.txt
python controller.py
```

**See [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed setup instructions.**

---

## Architecture

- **Audio Capture:** soundcard (system audio + microphone)
- **Transcription:** faster-whisper (local, offline)
- **Topic Analysis:** Parallax LLM (local)
- **Web UI:** Flask with live updates
- **Deployment:** Docker + GitHub Container Registry

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
# Audio
SAMPLE_RATE=16000
MODEL_NAME=small
MIC_NAME_FILTER=Yeti

# Parallax LLM
PARALLAX_ENDPOINT=http://localhost:3001/v1
SUMMARY_INTERVAL_SECONDS=60
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for all configuration options.

---

## Hackathon Multi-Computer Setup

### Control Computer
```bash
docker-compose -f docker-compose.control.yml up -d
```
Runs Parallax scheduler + convo-mapper

### Worker Computers
```bash
export SCHEDULER_ADDRESS=192.168.1.100  # Control computer IP
docker-compose -f docker-compose.worker.yml up -d
```
Joins Parallax cluster as worker

---

## Development

### Setup
```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Code Quality
```bash
# Run all checks
pre-commit run --all-files

# Format code
black . --line-length 100
isort . --profile black
```

### Docker Build
```bash
docker build -t convo-mapper:dev .
docker run -p 5000:5000 convo-mapper:dev
```

---

## CI/CD Pipeline

Automated workflows via GitHub Actions:

- **Pre-commit Checks** - Code quality (black, isort, ruff)
- **CI Tests** - Cross-platform testing (Ubuntu/macOS, Python 3.11/3.12)
- **Docker Build** - Automatic image build and push to GHCR

See [CI/CD_Plan.md](CI/CD_Plan.md) for detailed pipeline documentation.

---

## Documentation

- **[QUICKSTART.md](docs/QUICKSTART.md)** - Setup guide for teammates
- **[CI/CD_Plan.md](CI/CD_Plan.md)** - Complete CI/CD implementation plan
- **[CLAUDE.md](CLAUDE.md)** - Project overview and architecture

---

## Privacy & Security

🔒 **Your conversations never leave your computer**

- Transcription runs locally (faster-whisper)
- LLM runs locally (Parallax)
- No data sent to external servers
- Optional: In-memory mode (no disk storage)

---

## System Requirements

- **OS:** macOS, Linux, or Windows (WSL)
- **Python:** 3.11 or 3.12
- **Memory:** 4GB+ RAM
- **Audio:** Microphone access required
- **Optional:** GPU for faster transcription (CUDA support)

---

## Troubleshooting

### Docker audio not working
**Linux:** Check PulseAudio configuration
**macOS:** Use native installation (Docker audio not supported)

### Can't connect to Parallax
Verify Parallax is running: `curl http://localhost:3001/v1/models`

### Wrong microphone selected
Find your mic: `python -c "import soundcard as sc; [print(m.name) for m in sc.all_microphones()]"`

**See [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed troubleshooting.**

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

Built for the Parallax hackathon project.
