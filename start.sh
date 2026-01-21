#!/bin/bash
# Start FastAPI in the background
uvicorn api.app:app --host 0.0.0.0 --port 8000 &

# Wait 15 seconds for the model to load
sleep 15

# Start Streamlit
streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0