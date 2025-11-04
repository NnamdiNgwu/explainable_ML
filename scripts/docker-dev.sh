#!/bin/bash

# Development
echo "🔧 Starting Explainable AI (Development)..."
docker-compose -f docker-compose.dev.yml up

# Wait for health check
sleep 5
curl http://localhost:5000/api/v1/health