#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh  —  Launch FastAPI backend + Streamlit frontend
# Run from the flood_predictor/ root directory:
#   chmod +x start.sh && ./start.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

# Resolve the directory this script lives in (flood_predictor/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

echo "🌊  Flood Prediction System — Starting up"
echo "────────────────────────────────────────────"

# ── OpenWeatherMap API key ─────────────────────────────────────────────────
if [ -z "$OWM_API_KEY" ]; then
    echo "⚠️   OWM_API_KEY is not set — rainfall will use dataset fallback."
    echo "    To enable live rainfall:  export OWM_API_KEY='your_key_here'"
    echo ""
else
    echo "🌧️   OWM_API_KEY is set — live rainfall enabled."
fi
echo "────────────────────────────────────────────"

# ── Activate virtualenv ───────────────────────────────────────────────────────
# Looks for flood_env first (your env), then .venv as a fallback.
if [ -f "$SCRIPT_DIR/flood_env/bin/activate" ]; then
    echo "🐍  Activating flood_env ..."
    source "$SCRIPT_DIR/flood_env/bin/activate"
elif [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    echo "🐍  Activating .venv ..."
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    echo "⚠️   No virtualenv found at flood_env/ or .venv/ — using system Python."
    echo "    To create one:"
    echo "      python3 -m venv flood_env"
    echo "      source flood_env/bin/activate"
    echo "      pip install -r requirements.txt"
    echo ""
fi

# ── Verify Python is available ────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "❌  python3 not found."
    exit 1
fi

echo ""
echo "🚀  Starting FastAPI backend on http://localhost:8000 ..."
# Run uvicorn from inside backend/ so relative paths (../models, ../data) resolve correctly.
# The sys.path fix inside main.py handles `from src.x import y` automatically —
# no need to copy src/ anywhere.
cd "$SCRIPT_DIR/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# Give FastAPI a moment to finish startup before Streamlit tries to call it
sleep 3

echo ""
echo "🖥️   Starting Streamlit frontend on http://localhost:8501 ..."
cd "$SCRIPT_DIR/frontend"
streamlit run app.py --server.port 8501 --server.headless true &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo ""
echo "────────────────────────────────────────────"
echo "✅  Both services started."
echo "   Backend  → http://localhost:8000"
echo "   API Docs → http://localhost:8000/docs"
echo "   Frontend → http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both."
echo "────────────────────────────────────────────"

# Trap Ctrl+C / SIGTERM to cleanly kill both processes
trap "echo ''; echo 'Shutting down ...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait
