#!/bin/bash

# Open-WebUI Development Environment Startup Script
# Frontend: http://localhost:5173
# Backend: http://localhost:8080

set -e

echo "🚀 Starting Open-WebUI Development Environment..."
echo ""
echo "📦 Frontend will run at: http://localhost:5173"
echo "🔧 Backend will run at: http://localhost:8080"
echo ""

# Stop and remove old containers (if exists)
echo "🧹 Cleaning up old containers..."
docker compose -f docker-compose.dev.yaml down 2>/dev/null || true
docker stop reasoning-lens 2>/dev/null || true
docker rm reasoning-lens 2>/dev/null || true

# Ensure volume exists
echo "💾 Checking data volume..."
docker volume inspect reasoning-lens-data >/dev/null 2>&1 || docker volume create reasoning-lens-data

# Start development environment
echo "🎬 Starting development containers..."
docker compose -f docker-compose.dev.yaml up -d

echo ""
echo "✅ Development environment started!"
echo ""
echo "📋 Useful commands:"
echo "  View logs:          docker compose -f docker-compose.dev.yaml logs -f"
echo "  View backend logs:  docker compose -f docker-compose.dev.yaml logs -f backend"
echo "  View frontend logs: docker compose -f docker-compose.dev.yaml logs -f frontend"
echo "  Stop services:      docker compose -f docker-compose.dev.yaml down"
echo "  Restart backend:    docker compose -f docker-compose.dev.yaml restart backend"
echo "  Restart frontend:   docker compose -f docker-compose.dev.yaml restart frontend"
echo ""
echo "🌐 Access URLs:"
echo "  Frontend: http://localhost:5173"
echo "  Backend: http://localhost:8080"
echo ""

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 3

# Automatically show logs
echo "📄 Showing service logs (press Ctrl+C to stop):"
echo ""
docker compose -f docker-compose.dev.yaml logs -f

