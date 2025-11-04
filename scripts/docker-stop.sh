#!/bin/bash

echo "🛑 Stopping containers..."
docker-compose down
docker-compose -f docker-compose.dev.yml down

echo "✅ Stopped"