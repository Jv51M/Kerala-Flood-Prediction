# 🌊 Flood Prediction System
**Real-time flood risk assessment — Thrissur, Kerala**

Uses three ML models (Random Forest, LSTM, GNN) with live rainfall data from Open-Meteo
to predict flood risk at any geographic location within the dataset's coverage area.

---

## Project Structure

```
flood_predictor/
├── backend/
│   ├── main.py                  ← FastAPI app (all endpoints)
│   ├── lstm_model_train.py      ← LSTM training (saves scalers)
│   └── gnn_model_train.py       ← GNN training (saves scaler)
├── frontend/
│   └── app.py                   ← Streamlit UI
├── src/
│   ├── spatial.py               ← Haversine filtering, graph building
│   ├── rainfall_fetcher.py      ← Open-Meteo API calls
│   ├── rf_predictor.py          ← RF inference
│   ├── lstm_predictor.py        ← LSTM inference
│   └── gnn_predictor.py         ← GNN inference
├── models/                      ← Trained model files go here
│   ├── flood_rf_model.pkl
│   ├── flood_lstm_model.keras
│   ├── flood_gnn_model.pth
│   ├── scaler_temporal.pkl      ← saved by lstm_model_train.py
│   ├── scaler_static.pkl        ← saved by lstm_model_train.py
│   └── scaler_gnn.pkl           ← saved by gnn_model_train.py
├── data/
│   ├── randomforest-dataset.csv
│   ├── lstm_training_data.csv
│   └── gnn_training_data.csv
├── requirements.txt
└── start.sh                     ← one-command startup
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models & save scalers
```bash
# From the project root:
python backend/lstm_model_train.py   # → models/flood_lstm_model.keras + scalers
python backend/gnn_model_train.py    # → models/flood_gnn_model.pth + scaler

# RF model (your existing script, make sure it saves to models/flood_rf_model.pkl)
python scripts/randomforest.py
```

### 3. Start the system
```bash
chmod +x start.sh
./start.sh
```

Or start each service manually:
```bash
# Terminal 1 — Backend
cd backend && cp -r ../src src/
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
streamlit run app.py --server.port 8501
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System health + model status |
| POST | `/predict` | **Main prediction endpoint** |
| GET | `/nearby-points` | Inspect points within radius |
| GET | `/rainfall-history` | Raw 15-day rainfall for a coordinate |

### POST /predict — Request body
```json
{
  "latitude":  10.5276,
  "longitude": 76.2144,
  "radius_km": 3.0
}
```

### POST /predict — Response
```json
{
  "ensemble_probability": 0.72,
  "flood_risk": true,
  "risk_level": "HIGH",
  "nearby_point_count": 14,
  "rainfall_today_mm": 22.4,
  "random_forest": { "mean_probability": 0.68, "prediction": true,  "point_count": 14 },
  "lstm":          { "mean_probability": 0.75, "prediction": true,  "point_count": 14 },
  "gnn":           { "mean_probability": 0.73, "prediction": true,  "point_count": 14 }
}
```

---

## How It Works

```
User (lat, lon)
      │
      ▼
 Haversine filter
 on 126k-row RF dataset
      │
      ▼
 Nearby points (terrain: elevation, slope, dist_to_river, …)
      │
      ├──────────────────────────────────────┐
      ▼                                      ▼
 Open-Meteo API                      Use point coords
 15-day daily rainfall               for spatial graph (GNN)
 per nearby point
      │
      ├──────────┬──────────────┬────────────┘
      ▼          ▼              ▼
  Random       LSTM            GNN
  Forest    (15-day seq     (graph of
  (today's   + terrain)      nearby pts)
  rain +
  terrain)
      │          │              │
      └──────────┴──────────────┘
                 │
                 ▼
          Ensemble average
          → Risk level + probability
```

---

## Risk Levels

| Level | Probability |
|-------|-------------|
| 🟢 LOW      | 0 – 25% |
| 🟠 MODERATE | 25 – 50% |
| 🔴 HIGH     | 50 – 75% |
| 🟣 EXTREME  | 75 – 100% |

---

## Notes

- **No API key required** — rainfall data from [Open-Meteo](https://open-meteo.com) is free.
- If scalers are missing, the system falls back to z-score normalisation on the incoming data.
  This is less accurate — always save scalers during training.
- The GNN rebuilds its spatial graph at inference time using only the nearby subset of points,
  so predictions are spatially local rather than global.
