# CI/CD Pipeline Implementation Plan for Convo-Mapper

## Overview
Set up a complete CI/CD pipeline for the convo-mapper hackathon project using GitHub Actions, with Docker-based deployment for easy distribution across teammate computers.

## Current State Analysis
- **No CI/CD**: No workflows, no automated testing
- **No tests**: No pytest setup, no test files
- **Unpinned dependencies**: requirements.txt has no version constraints
- **Hardcoded configs**: All settings in main.py and loop.py constants
- **Flat structure**: No organized directories for tests, configs, etc.
- **NumPy patch**: Critical compatibility shim at main.py:12-28

## Recommended Approach

### Deployment Strategy: Docker Containers
**Rationale for hackathon context:**
- Ensures consistent environment across control + worker computers
- Bundles all dependencies (faster-whisper, audio drivers, Flask)
- Easy distribution via GitHub Container Registry
- Avoids "works on my machine" issues during demo
- Simplifies Parallax LLM integration

**Trade-off:** Docker audio device access requires configuration on Linux, but provides best team experience overall.

## Implementation Plan (Prioritized for Hackathon)

### Phase 0: Setup & Documentation (30 min) - Prerequisites
**Goal:** Prepare git workflow and document the plan

#### 0.1 Create Plan Documentation (15 min)
**File:** `CI/CD_Plan.md` (new)
- Copy the comprehensive plan to a markdown file in the repository
- This serves as reference documentation for the team
- Include all phases, timelines, and key decisions

#### 0.2 Create Feature Branch (15 min)
**Git workflow:**
```bash
# Create local branch
git checkout -b CI/CD_dev

# Push to remote and set upstream
git push -u origin CI/CD_dev
```

This branch will contain all CI/CD implementation work and can be merged to main once tested.

**Phase 0 Deliverable:** Plan documented and feature branch ready for development

---

### Phase 1: Foundation & Distribution (Day 1) - Critical Path
**Goal:** Enable teammates to pull and run containerized application

#### 1.1 Pin Dependencies (30 min)
**File:** `requirements.txt`
```txt
faster-whisper==1.1.0
soundcard==0.4.3
flask==3.0.0
numpy==2.1.0
pandas==2.2.0
requests==2.31.0
openai==1.12.0
```

**File:** `requirements-dev.txt` (new)
```txt
pytest==8.2.0
pytest-cov==5.0.0
pytest-mock==3.14.0
black==24.3.0
isort==5.13.2
ruff==0.4.0
pre-commit==3.7.0
```

#### 1.2 Code Quality Setup (45 min)
**File:** `.pre-commit-config.yaml` (new)
- Match parent Parallax project standards
- Tools: black (line-length 100), isort (profile: black), ruff, autoflake

**File:** `pyproject.toml` (new)
- Tool configurations for black, isort, ruff
- Pytest configuration and markers
- Coverage settings

**Action:** Run `pre-commit run --all-files` and fix formatting issues

#### 1.3 Configuration Management (1 hour)
**File:** `config.py` (new)
- Centralize all configuration
- Load from environment variables with defaults
- Replace hardcoded constants in main.py and loop.py

**Refactor:** Update main.py and loop.py to import from config.py

#### 1.4 Dockerfile (1 hour)
Create multi-stage Dockerfile with audio dependencies

#### 1.5 Docker Compose for Team Distribution (1 hour)
- `docker-compose.yml` - Standard deployment
- `docker-compose.control.yml` - For control computer (Parallax + convo-mapper)
- `docker-compose.worker.yml` - For worker computers (Parallax worker)
- `.env.example` - Configuration template

#### 1.6 GitHub Actions - Docker Build (1 hour)
Automated Docker image building and push to GitHub Container Registry

#### 1.7 Quick Start Documentation (1 hour)
- `docs/QUICKSTART.md` - Team onboarding guide
- `scripts/quick-start.sh` - Quick start script

**Phase 1 Deliverable:** Teammates can `docker-compose pull && docker-compose up -d` and have a running system

---

### Phase 2: Testing & Quality (Day 2) - Stability
**Goal:** Prevent regressions and enable confident iteration

#### 2.1 Test Infrastructure Setup (30 min)
Create test directory structure with pytest configuration

#### 2.2 Priority Tests (2.5 hours)
**Must-have tests:**
1. test_numpy_patch.py - Compatibility shim
2. test_flask_routes.py - API endpoints
3. test_chunker.py - Audio accumulation
4. test_audio_capture.py - Mock soundcard
5. test_transcriber.py - Mock WhisperModel
6. test_topic_loop.py - Mock OpenAI client

**Target:** >50% coverage on critical paths

#### 2.3 GitHub Actions - CI Workflow (1 hour)
Automated testing on Ubuntu and macOS with Python 3.11/3.12

#### 2.4 Pre-commit Workflow (30 min)
Code quality checks on every PR

**Phase 2 Deliverable:** Automated testing catches bugs before production

---

### Phase 3: Documentation & Release (Day 3) - Polish
**Goal:** Make it easy for teammates and enable versioned releases

#### 3.1 Core Documentation (1.5 hours)
- README.md updates (badges, quick start)
- CONTRIBUTING.md (development setup, guidelines)
- docs/ARCHITECTURE.md (system components, data flow)

#### 3.2 Release Workflow (30 min)
Automated GitHub releases with semantic versioning

**Phase 3 Deliverable:** Professional presentation ready for hackathon demo

---

## Critical Files to Modify

### Must Create
1. `.github/workflows/docker-build.yml` - Core distribution mechanism
2. `.github/workflows/ci.yml` - Automated testing
3. `.github/workflows/pre-commit.yml` - Code quality
4. `Dockerfile` - Container definition
5. `docker-compose.yml` - Easy deployment
6. `config.py` - Centralized configuration
7. `.pre-commit-config.yaml` - Local code quality
8. `pyproject.toml` - Tool configurations
9. `requirements-dev.txt` - Dev dependencies
10. `tests/` directory structure
11. `docs/QUICKSTART.md` - Team onboarding

### Must Modify
1. `requirements.txt` - Pin exact versions
2. `main.py` - Import from config.py
3. `loop.py` - Import from config.py
4. `README.md` - Add CI/CD badges and Docker instructions

---

## Key Technical Decisions

### 1. Docker over Native Installation
**Rationale:** Hackathon distributed setup, consistency, simplifies Parallax integration
**Trade-off:** Linux audio device access requires configuration

### 2. GitHub Container Registry over Docker Hub
**Rationale:** Free for public repos, better GitHub integration, no rate limiting

### 3. network_mode: host in Docker Compose
**Rationale:** Simplifies Parallax localhost access, faster setup
**Trade-off:** Less network isolation

### 4. Centralized config.py
**Rationale:** Environment variable configuration (12-factor app), easier testing

### 5. Mocked Tests over Real Hardware
**Rationale:** CI runners lack audio devices, faster execution, consistent environment

---

## Hackathon-Specific Considerations

### Multi-Computer Setup
**Control Computer:**
```bash
docker-compose -f docker-compose.control.yml up -d
# Runs: Parallax scheduler + convo-mapper
```

**Worker Computers:**
```bash
export SCHEDULER_ADDRESS=192.168.1.100  # Control computer IP
docker-compose -f docker-compose.worker.yml up -d
# Runs: Parallax worker
```

### Rapid Iteration During Hackathon
1. Make code changes
2. Git push → triggers Docker build
3. Teammates: `docker-compose pull && docker-compose restart`
4. New version in ~2 minutes

### Fallback Plans
- macOS audio issues → Run natively
- Docker not working → Git distribution
- No GPU → Smaller Whisper model

---

## Risk Mitigation

1. **Audio devices in Docker**: Extensive PulseAudio docs, troubleshooting, fallback
2. **NumPy 2.x patch**: Pin numpy==2.1.0, test coverage, documentation
3. **Slow CI tests**: Mock WhisperModel, cache dependencies
4. **Parallax connection**: network_mode: host, troubleshooting docs

---

## Success Metrics

### Phase 1 (Critical)
- [ ] Docker image builds successfully
- [ ] Image pushed to GHCR automatically
- [ ] Teammate can pull and run
- [ ] Web UI accessible
- [ ] Audio works on Linux

### Phase 2
- [ ] >50% test coverage
- [ ] CI passes on Ubuntu/macOS
- [ ] Pre-commit hooks working
- [ ] Consistent formatting

### Phase 3
- [ ] Clear onboarding docs
- [ ] Versioned releases (v0.1.0)
- [ ] Passing build badges
- [ ] Quick-start script works

---

## Implementation Timeline

**Day 1 (4-6h):** Phase 1 - Foundation & Distribution
**Day 2 (4-6h):** Phase 2 - Testing & Quality
**Day 3 (2-3h):** Phase 3 - Documentation & Release
**Total:** 10-15 hours over 3 days
