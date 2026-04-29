#!/usr/bin/env bash
# start.sh — Launch DeepDiligence backend + frontend in one command
#
# Usage:
#   ./start.sh            # starts both backend (port 8000) and frontend (port 8080)
#   ./start.sh --backend  # backend only
#   ./start.sh --frontend # frontend only

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

start_backend() {
  echo "🐍  Starting FastAPI backend on http://localhost:8000 ..."
  cd "$ROOT"
  python3.11 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
}

start_frontend() {
  echo "⚛️   Starting React frontend on http://localhost:8080 ..."
  cd "$ROOT/frontend"
  npm run dev
}

case "${1:-both}" in
  --backend)
    start_backend
    ;;
  --frontend)
    start_frontend
    ;;
  *)
    # Run both concurrently; trap Ctrl-C to kill both
    trap 'kill 0' INT TERM

    start_backend &
    BACKEND_PID=$!

    sleep 2  # give FastAPI a moment to bind its port
    start_frontend &
    FRONTEND_PID=$!

    echo ""
    echo "✅  DeepDiligence is running:"
    echo "   Backend  →  http://localhost:8000"
    echo "   Frontend →  http://localhost:8080"
    echo ""
    echo "   Press Ctrl-C to stop both."

    wait $BACKEND_PID $FRONTEND_PID
    ;;
esac
