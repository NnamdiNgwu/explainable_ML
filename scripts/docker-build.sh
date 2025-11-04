#!/bin/bash
set -e

echo "🐳 Building Explainable AI Docker image..."
docker build -t explainable-ai:latest .
docker build -t explainable-ai:dev -f Dockerfile.dev .

echo "✅ Build complete!"
echo ""
echo "To run development:"
echo "  docker-compose -f docker-compose.dev.yml up"