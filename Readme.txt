# 🎭 Advanced Sentiment Analysis with Enhanced DistilBERT

Production-ready sentiment analysis achieving **95.17% accuracy** on SST-2 dataset.

## Features

- 🚀 Enhanced DistilBERT with LoRA adapters
- 📊 95.17% validation accuracy
- ⚡ Fast inference (<50ms)
- 🎯 Knowledge distillation from BERT
- 🔧 Production-ready deployment

## Quick Start

### Installation
\`\`\`bash
git clone https://github.com/yourusername/sentiment-analysis-distilbert
cd sentiment-analysis-distilbert
pip install -r requirements.txt
\`\`\`

### Run API
\`\`\`bash
uvicorn api.app:app --reload --port 8000
\`\`\`

### Run Streamlit App
\`\`\`bash
streamlit run streamlit_app.py
\`\`\`

## Model Performance

- **Dataset:** SST-2 (67,349 samples)
- **Accuracy:** 95.17%
- **Architecture:** DistilBERT + LoRA
- **Parameters:** 66M total, 4M trainable
- **Training Time:** 2.5 hours on Kaggle P100

## Deployment

- 🌐 **Live Demo:** [HuggingFace Spaces](#)
- 📚 **API Docs:** [FastAPI Docs](http://localhost:8000/docs)
- 🐳 **Docker:** `docker-compose up`

## Project Structure

\`\`\`
sentiment-analysis-distilbert/
├── models/               # Trained model
├── src/                  # Model architecture & inference
├── api/                  # FastAPI application
├── streamlit_app.py      # Streamlit frontend
├── requirements.txt
└── README.md
\`\`\`

