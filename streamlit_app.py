"""
🎭 Production Streamlit App - Advanced Sentiment Analysis
Complete end-to-end interface using FastAPI backend

File: streamlit_app.py
Place this in the root folder of your project
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json

# ================================
# CONFIGURATION
# ================================
API_URL = "http://localhost:8000"  # Your FastAPI backend

st.set_page_config(
    page_title="Sentiment Analysis - Enhanced DistilBERT",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .positive-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .negative-result {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 10px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


# ================================
# API FUNCTIONS
# ================================
@st.cache_data(ttl=300)
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

@st.cache_data(ttl=300)
def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def predict_sentiment(text):
    """Get sentiment prediction from API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def predict_batch(texts):
    """Get batch predictions from API"""
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"texts": texts},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Batch Error: {str(e)}")
        return None

def get_examples():
    """Get example predictions"""
    try:
        response = requests.get(f"{API_URL}/examples", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


# ================================
# INITIALIZE SESSION STATE
# ================================
if 'history' not in st.session_state:
    st.session_state.history = []

if 'api_status' not in st.session_state:
    st.session_state.api_status = check_api_health()


# ================================
# HEADER
# ================================
st.markdown('<h1 class="main-header">🎭 Advanced Sentiment Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Enhanced DistilBERT with LoRA Adapters</p>', unsafe_allow_html=True)

# API Status Check
if not st.session_state.api_status:
    st.error("⚠️ API is not running! Please start the FastAPI backend:")
    st.code("uvicorn api.app:app --reload --port 8000", language="bash")
    st.stop()


# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=100)
    st.title("📊 Model Information")
    
    # Get model info
    model_info = get_model_info()
    
    if model_info:
        st.metric("Model Accuracy", model_info.get('accuracy', 'N/A'))
        st.metric("Training Epoch", model_info.get('epoch', 'N/A'))
        
        st.markdown("---")
        
        st.markdown("### 🔧 Model Details")
        st.markdown(f"**Total Parameters:** {model_info.get('total_parameters', 'N/A')}")
        st.markdown(f"**Trainable Params:** {model_info.get('trainable_parameters', 'N/A')}")
        st.markdown(f"**Efficiency:** {model_info.get('efficiency', 'N/A')}")
        st.markdown(f"**Device:** {model_info.get('device', 'N/A')}")
    
    st.markdown("---")
    
    st.markdown("### ✨ Features")
    features = [
        "🚀 LoRA Adapters (6% params)",
        "📊 R-Drop Regularization",
        "🎯 Knowledge Distillation",
        "⚡ Fast Inference (<50ms)",
        "🔒 Production Ready"
    ]
    for feature in features:
        st.markdown(feature)
    
    st.markdown("---")
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


# ================================
# MAIN CONTENT
# ================================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Prediction", "📦 Batch Analysis", "📈 Examples", "📚 About"])

# ================================
# TAB 1: SINGLE PREDICTION
# ================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter your text")
        user_text = st.text_area(
            "Type or paste text here...",
            height=200,
            placeholder="e.g., This movie was absolutely fantastic! The acting was superb.",
            label_visibility="collapsed"
        )
        
        analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        
        if analyze_btn and user_text.strip():
            with st.spinner("Analyzing sentiment..."):
                result = predict_sentiment(user_text)
                
                if result:
                    # Add to history
                    st.session_state.history.insert(0, {
                        'text': user_text[:100] + '...' if len(user_text) > 100 else user_text,
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence'],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Display result
                    if result['sentiment'] == 'positive':
                        st.markdown(
                            f'<div class="positive-result">😊 POSITIVE</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="negative-result">😞 NEGATIVE</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    with col_b:
                        st.metric("Label", result['label'])
                    with col_c:
                        st.metric("Time (ms)", f"{result['inference_time_ms']:.1f}")
                    
                    # Probability chart
                    st.markdown("### Probability Distribution")
                    probs = result['probabilities']
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Negative', 'Positive'],
                            y=[probs['negative'], probs['positive']],
                            marker=dict(
                                color=['#eb3349', '#38ef7d'],
                                line=dict(color='white', width=2)
                            ),
                            text=[f"{probs['negative']*100:.1f}%", f"{probs['positive']*100:.1f}%"],
                            textposition='auto',
                            textfont=dict(size=16, color='white', family='Arial Black')
                        )
                    ])
                    
                    fig.update_layout(
                        title="Sentiment Probabilities",
                        yaxis_title="Probability",
                        yaxis=dict(range=[0, 1], tickformat='.0%'),
                        height=400,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        elif analyze_btn:
            st.warning("⚠️ Please enter some text to analyze")
    
    with col2:
        st.markdown("### 📝 Quick Examples")
        examples = [
            "This movie was absolutely fantastic!",
            "Worst experience ever. Completely disappointed.",
            "It was okay, nothing special.",
            "Amazing product! Highly recommended!",
            "Terrible customer service."
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Try: {example[:30]}...", key=f"ex_{i}"):
                user_text = example
                st.rerun()
        
        # Recent history
        if st.session_state.history:
            st.markdown("### 📜 Recent Predictions")
            for i, item in enumerate(st.session_state.history[:5]):
                emoji = "😊" if item['sentiment'] == 'positive' else "😞"
                st.markdown(f"""
                <div class="feature-box">
                    <small>{item['timestamp']}</small><br>
                    {emoji} <b>{item['sentiment'].upper()}</b> ({item['confidence']*100:.0f}%)<br>
                    <small>{item['text']}</small>
                </div>
                """, unsafe_allow_html=True)


# ================================
# TAB 2: BATCH ANALYSIS
# ================================
with tab2:
    st.markdown("### 📦 Batch Sentiment Analysis")
    st.info("Analyze multiple texts at once. Enter one text per line (max 100 texts).")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        batch_input = st.text_area(
            "Enter multiple texts (one per line):",
            height=300,
            placeholder="This is great!\nThis is terrible!\nI love it!",
            label_visibility="collapsed"
        )
        
        if st.button("🚀 Analyze Batch", type="primary", use_container_width=True):
            if batch_input.strip():
                texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
                
                if len(texts) > 100:
                    st.error("⚠️ Maximum 100 texts allowed per batch")
                else:
                    with st.spinner(f"Analyzing {len(texts)} texts..."):
                        batch_result = predict_batch(texts)
                        
                        if batch_result and 'results' in batch_result:
                            results = batch_result['results']
                            
                            # Summary metrics
                            st.markdown("### 📊 Summary")
                            col_a, col_b, col_c, col_d = st.columns(4)
                            
                            positive_count = sum(1 for r in results if r['sentiment'] == 'positive')
                            negative_count = len(results) - positive_count
                            avg_confidence = sum(r['confidence'] for r in results) / len(results)
                            avg_time = sum(r['inference_time_ms'] for r in results) / len(results)
                            
                            with col_a:
                                st.metric("Total Texts", len(results))
                            with col_b:
                                st.metric("Positive", positive_count)
                            with col_c:
                                st.metric("Negative", negative_count)
                            with col_d:
                                st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
                            
                            # Results table
                            st.markdown("### 📋 Detailed Results")
                            df = pd.DataFrame([
                                {
                                    'Text': r['text'][:50] + '...' if len(r['text']) > 50 else r['text'],
                                    'Sentiment': r['sentiment'].upper(),
                                    'Confidence': f"{r['confidence']*100:.1f}%",
                                    'Negative Prob': f"{r['probabilities']['negative']*100:.1f}%",
                                    'Positive Prob': f"{r['probabilities']['positive']*100:.1f}%",
                                }
                                for r in results
                            ])
                            
                            st.dataframe(df, use_container_width=True, height=400)
                            
                            # Download button
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
            else:
                st.warning("⚠️ Please enter some texts to analyze")
    
    with col2:
        st.markdown("### 📊 Visualization")
        
        # Placeholder chart (will update after analysis)
        if 'batch_result' in locals() and batch_result:
            results = batch_result['results']
            
            # Pie chart
            sentiments = [r['sentiment'] for r in results]
            sentiment_counts = pd.Series(sentiments).value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                marker=dict(colors=['#38ef7d', '#eb3349']),
                textfont=dict(size=16, color='white')
            )])
            
            fig.update_layout(
                title="Sentiment Distribution",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence distribution
            confidences = [r['confidence'] for r in results]
            
            fig2 = go.Figure(data=[go.Histogram(
                x=confidences,
                nbinsx=20,
                marker=dict(color='#667eea')
            )])
            
            fig2.update_layout(
                title="Confidence Distribution",
                xaxis_title="Confidence",
                yaxis_title="Count",
                height=300
            )
            
            st.plotly_chart(fig2, use_container_width=True)


# ================================
# TAB 3: EXAMPLES
# ================================
with tab3:
    st.markdown("### 📈 Pre-computed Examples")
    st.info("See how the model performs on various example texts")
    
    if st.button("🔄 Load Examples", use_container_width=True):
        with st.spinner("Loading examples..."):
            examples_data = get_examples()
            
            if examples_data and 'examples' in examples_data:
                examples = examples_data['examples']
                
                for i, example in enumerate(examples):
                    with st.expander(f"Example {i+1}: {example['text'][:60]}..."):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Text:** {example['text']}")
                            
                            if example['sentiment'] == 'positive':
                                st.success(f"😊 **POSITIVE** (Confidence: {example['confidence']*100:.1f}%)")
                            else:
                                st.error(f"😞 **NEGATIVE** (Confidence: {example['confidence']*100:.1f}%)")
                        
                        with col2:
                            probs = example['probabilities']
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=['Neg', 'Pos'],
                                    y=[probs['negative'], probs['positive']],
                                    marker=dict(color=['#eb3349', '#38ef7d'])
                                )
                            ])
                            
                            fig.update_layout(
                                height=200,
                                showlegend=False,
                                margin=dict(l=20, r=20, t=20, b=20)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)


# ================================
# TAB 4: ABOUT
# ================================
with tab4:
    st.markdown("### 📚 About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Model Architecture
        
        This sentiment analysis system uses an **Enhanced DistilBERT** model with several advanced features:
        
        - **Base Model:** DistilBERT-base-uncased
        - **Enhancement:** LoRA (Low-Rank Adaptation) layers
        - **Training:** Knowledge distillation from BERT-base
        - **Regularization:** R-Drop for better generalization
        - **Dataset:** SST-2 (Stanford Sentiment Treebank)
        - **Samples:** 67,349 training samples
        
        #### 🚀 Key Features
        
        - **Parameter Efficient:** Only 6% of parameters are trainable
        - **High Accuracy:** 95.17% on validation set
        - **Fast Inference:** <50ms per prediction
        - **Production Ready:** FastAPI backend + Streamlit frontend
        - **Scalable:** Batch processing support
        
        #### 🔧 Technical Stack
        
        - **Backend:** FastAPI
        - **Frontend:** Streamlit
        - **ML Framework:** PyTorch
        - **Transformers:** HuggingFace
        - **Deployment:** Docker, HuggingFace Spaces, Vercel
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Performance Metrics
        """)
        
        # Performance visualization
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1-Score', 'Precision', 'Recall'],
            'Value': [95.17, 94.8, 95.2, 94.5]
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics_df['Metric'],
                y=metrics_df['Value'],
                marker=dict(
                    color=metrics_df['Value'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=metrics_df['Value'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Model Performance Metrics",
            yaxis_title="Score (%)",
            yaxis=dict(range=[0, 100]),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        #### 🎓 Training Details
        
        - **Epochs:** 5
        - **Batch Size:** 32
        - **Learning Rate:** 2e-5
        - **Optimizer:** AdamW
        - **Scheduler:** Cosine with warmup
        - **Training Time:** ~2.5 hours on Kaggle GPU
        
        #### 📈 Use Cases
        
        - **Customer Feedback Analysis**
        - **Social Media Monitoring**
        - **Product Review Classification**
        - **Brand Sentiment Tracking**
        - **Support Ticket Prioritization**
        
        #### 🔗 Links
        
        - [GitHub Repository](#)
        - [Model on HuggingFace](#)
        - [API Documentation](http://localhost:8000/docs)
        - [Research Paper](#)
        """)


# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><b>Advanced Sentiment Analysis with Enhanced DistilBERT</b></p>
    <p>Built with ❤️ using Streamlit, FastAPI, and PyTorch</p>
    <p>© 2024 | <a href="https://github.com/yourusername">GitHub</a> | <a href="http://localhost:8000/docs">API Docs</a></p>
</div>
""", unsafe_allow_html=True)