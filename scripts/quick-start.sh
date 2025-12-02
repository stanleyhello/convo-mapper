#!/bin/bash
set -e

# Convo-Mapper Quick Start Script
# This script helps you quickly set up and run convo-mapper

echo "========================================="
echo "  Convo-Mapper Quick Start"
echo "========================================="
echo ""

# Check if Docker is running
echo "Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "   Please start Docker Desktop and try again"
    exit 1
fi
echo "✓ Docker is running"
echo ""

# Check if .env exists, create from example if not
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✓ Created .env file"
        echo ""
        echo "⚠️  IMPORTANT: Please review and edit .env with your configuration:"
        echo "   - Set your microphone name (MIC_NAME_FILTER)"
        echo "   - Adjust model size if needed (MODEL_NAME)"
        echo "   - Configure Parallax endpoint if different"
        echo ""
        echo "After editing .env, run this script again."
        exit 0
    else
        echo "⚠️  Warning: .env.example not found, continuing without .env"
    fi
else
    echo "✓ Using existing .env configuration"
fi
echo ""

# Pull latest Docker images
echo "Pulling latest Docker images..."
if docker-compose pull; then
    echo "✓ Images pulled successfully"
else
    echo "⚠️  Warning: Could not pull images (you may need to authenticate)"
    echo "   Continuing with local build..."
fi
echo ""

# Start services
echo "Starting services..."
if docker-compose up -d; then
    echo "✓ Services started successfully!"
else
    echo "❌ Failed to start services"
    echo "   Check 'docker-compose logs' for details"
    exit 1
fi
echo ""

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 5

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✓ Services are running"
    echo ""
    echo "========================================="
    echo "  🎉 Setup Complete!"
    echo "========================================="
    echo ""
    echo "Web UI: http://localhost:5000"
    echo ""
    echo "Useful commands:"
    echo "  View logs:    docker-compose logs -f"
    echo "  Stop:         docker-compose down"
    echo "  Restart:      docker-compose restart"
    echo "  Status:       docker-compose ps"
    echo ""
else
    echo "⚠️  Services started but may not be running correctly"
    echo "   Check status: docker-compose ps"
    echo "   Check logs:   docker-compose logs -f"
fi
