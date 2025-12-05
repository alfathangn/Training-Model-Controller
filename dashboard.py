# dashboard_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import pickle
import os
import time

# ==================== KONFIGURASI ====================
BASE_DIR = r"C:\Users\LENOVO\OneDrive\Documents\A6 Arduino\Belajar\Trainingdht"
MODELS_DIR = os.path.join(BASE_DIR, "Trainingdht", "models_v2")

# ==================== SETUP PAGE ====================
st.set_page_config(
    page_title="DHT ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    """Load ML models dengan caching"""
    models = {}
    try:
        # Load scaler
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load models
        model_files = {
            'Decision Tree': 'decision_tree.pkl',
            'KNN': 'knn.pkl',
            'Logistic Regression': 'logistic_regression.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
        
        return models, scaler
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# ==================== FUNGSI PREDIKSI ====================
def predict_temperature(models, scaler, temperature, humidity, hour=None, minute=None):
    """Prediksi dengan semua model"""
    if hour is None or minute is None:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
    
    features = np.array([[temperature, humidity, hour, minute]])
    features_scaled = scaler.transform(features)
    
    predictions = {}
    
    for model_name, model in models.items():
        try:
            # Predict
            pred_code = model.predict(features_scaled)[0]
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(features_scaled)[0]
                confidence = probs[pred_code]
            else:
                confidence = 1.0
                probs = [0, 0, 0]
            
            # Map to label
            label_map = {0: 'DINGIN', 1: 'NORMAL', 2: 'PANAS'}
            label = label_map.get(pred_code, 'UNKNOWN')
            
            predictions[model_name] = {
                'label': label,
                'confidence': float(confidence),
                'probabilities': {
                    'DINGIN': float(probs[0]) if len(probs) > 0 else 0,
                    'NORMAL': float(probs[1]) if len(probs) > 1 else 0,
                    'PANAS': float(probs[2]) if len(probs) > 2 else 0
                },
                'label_encoded': int(pred_code),
                'color': get_label_color(label)
            }
            
        except Exception as e:
            predictions[model_name] = {
                'label': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    return predictions

def get_label_color(label):
    """Get color based on label"""
    colors = {
        'DINGIN': '#3498db',    # Blue
        'NORMAL': '#2ecc71',    # Green
        'PANAS': '#e74c3c',     # Red
        'UNKNOWN': '#95a5a6',   # Gray
        'ERROR': '#f39c12'      # Orange
    }
    return colors.get(label, '#95a5a6')

# ==================== GENERATE SAMPLE DATA ====================
def generate_sample_data():
    """Generate sample data untuk demo"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='H')
    
    data = []
    for date in dates:
        temp = np.random.normal(24, 5)  # Mean 24, std 5
        hum = np.random.normal(60, 15)  # Mean 60, std 15
        
        # Simple logic untuk label
        if temp < 22:
            label = 'DINGIN'
        elif temp > 26:
            label = 'PANAS'
        else:
            label = 'NORMAL'
        
        data.append({
            'timestamp': date,
            'temperature': round(temp, 1),
            'humidity': round(hum, 1),
            'label': label,
            'model': np.random.choice(['Decision Tree', 'KNN', 'Logistic Regression'])
        })
    
    return pd.DataFrame(data)

# ==================== SIDEBAR ====================
def sidebar_controls():
    """Sidebar controls"""
    st.sidebar.title("⚙️ Kontrol Dashboard")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    show_dt = st.sidebar.checkbox("Decision Tree", value=True)
    show_knn = st.sidebar.checkbox("K-Nearest Neighbors", value=True)
    show_lr = st.sidebar.checkbox("Logistic Regression", value=True)
    
    # Manual prediction
    st.sidebar.subheader("🔮 Manual Prediction")
    manual_temp = st.sidebar.slider("Temperature (°C)", 15.0, 35.0, 24.0, 0.5)
    manual_hum = st.sidebar.slider("Humidity (%)", 30.0, 90.0, 65.0, 1.0)
    manual_hour = st.sidebar.slider("Hour", 0, 23, datetime.now().hour)
    manual_minute = st.sidebar.slider("Minute", 0, 59, datetime.now().minute)
    
    # Time range
    st.sidebar.subheader("📅 Time Range")
    days_back = st.sidebar.slider("Days to display", 1, 30, 7)
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    return {
        'models': {'DT': show_dt, 'KNN': show_knn, 'LR': show_lr},
        'manual_input': (manual_temp, manual_hum, manual_hour, manual_minute),
        'days_back': days_back
    }

# ==================== MAIN DASHBOARD ====================
def main():
    # Header
    st.title("🤖 DHT Machine Learning Dashboard")
    st.markdown("Real-time temperature prediction with 3 ML models")
    st.markdown("---")
    
    # Load models
    models, scaler = load_models()
    
    if models is None or scaler is None:
        st.error("❌ Failed to load ML models. Please run training first.")
        return
    
    # Sidebar controls
    controls = sidebar_controls()
    show_models = controls['models']
    manual_input = controls['manual_input']
    
    # Row 1: Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌡️ Temperature", f"{manual_input[0]}°C", "Input")
    
    with col2:
        st.metric("💧 Humidity", f"{manual_input[1]}%", "Input")
    
    with col3:
        st.metric("⏰ Time", f"{manual_input[2]:02d}:{manual_input[3]:02d}")
    
    with col4:
        now = datetime.now()
        st.metric("🕒 Current", now.strftime("%H:%M"))
    
    st.markdown("---")
    
    # Row 2: Manual Prediction Results
    st.subheader("🔮 Manual Prediction Results")
    
    # Make prediction
    predictions = predict_temperature(
        models, scaler, 
        manual_input[0], manual_input[1],
        manual_input[2], manual_input[3]
    )
    
    # Display predictions in columns
    pred_cols = st.columns(3)
    
    for idx, (model_name, pred) in enumerate(predictions.items()):
        with pred_cols[idx]:
            color = pred.get('color', '#95a5a6')
            
            # Card-like display
            st.markdown(f"""
            <div style="
                background-color: {color}20;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid {color};
                margin-bottom: 20px;
            ">
                <h3 style="color: {color}; margin-top: 0;">{model_name}</h3>
                <h1 style="color: {color}; font-size: 2.5em; margin: 10px 0;">
                    {pred['label']}
                </h1>
                <p style="font-size: 1.2em; margin: 5px 0;">
                    Confidence: <strong>{pred['confidence']:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probabilities bar chart
            if 'probabilities' in pred:
                prob_df = pd.DataFrame({
                    'Class': list(pred['probabilities'].keys()),
                    'Probability': list(pred['probabilities'].values())
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Class', 
                    y='Probability',
                    color='Class',
                    color_discrete_map={
                        'DINGIN': '#3498db',
                        'NORMAL': '#2ecc71',
                        'PANAS': '#e74c3c'
                    },
                    range_y=[0, 1]
                )
                fig.update_layout(
                    showlegend=False,
                    height=200,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Row 3: Model Comparison
    st.subheader("📊 Model Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison (sample data)
        accuracy_data = pd.DataFrame({
            'Model': ['Decision Tree', 'KNN', 'Logistic Regression'],
            'Accuracy': [0.92, 0.88, 0.85],
            'F1-Score': [0.91, 0.87, 0.84]
        })
        
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=accuracy_data['Model'], y=accuracy_data['Accuracy']),
            go.Bar(name='F1-Score', x=accuracy_data['Model'], y=accuracy_data['F1-Score'])
        ])
        
        fig.update_layout(
            title="Model Performance",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confusion matrix summary
        st.markdown("**Model Agreement**")
        
        # Check if all models agree
        labels = [pred['label'] for pred in predictions.values()]
        agreement = len(set(labels)) == 1
        
        if agreement:
            st.success(f"✅ All models agree: **{labels[0]}**")
            st.balloons()
        else:
            st.warning(f"⚠️ Models disagree: {', '.join(set(labels))}")
        
        # Display agreement matrix
        agreement_matrix = pd.DataFrame(index=models.keys(), columns=models.keys())
        
        for m1 in models.keys():
            for m2 in models.keys():
                if m1 == m2:
                    agreement_matrix.loc[m1, m2] = "✓"
                else:
                    agree = predictions[m1]['label'] == predictions[m2]['label']
                    agreement_matrix.loc[m1, m2] = "✓" if agree else "✗"
        
        st.dataframe(agreement_matrix, use_container_width=True)
    
    st.markdown("---")
    
    # Row 4: Historical Data & Trends
    st.subheader("📈 Historical Data & Trends")
    
    # Generate sample historical data
    historical_data = generate_sample_data()
    
    # Filter by date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=controls['days_back'])
    filtered_data = historical_data[
        (historical_data['timestamp'] >= start_date) &
        (historical_data['timestamp'] <= end_date)
    ]
    
    tab1, tab2, tab3 = st.tabs(["Temperature Trend", "Humidity Trend", "Label Distribution"])
    
    with tab1:
        fig = px.line(
            filtered_data, 
            x='timestamp', 
            y='temperature',
            color='label',
            color_discrete_map={
                'DINGIN': '#3498db',
                'NORMAL': '#2ecc71',
                'PANAS': '#e74c3c'
            },
            title="Temperature Trend Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.scatter(
            filtered_data,
            x='temperature',
            y='humidity',
            color='label',
            color_discrete_map={
                'DINGIN': '#3498db',
                'NORMAL': '#2ecc71',
                'PANAS': '#e74c3c'
            },
            title="Temperature vs Humidity",
            hover_data=['timestamp']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        label_counts = filtered_data['label'].value_counts()
        fig = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            color=label_counts.index,
            color_discrete_map={
                'DINGIN': '#3498db',
                'NORMAL': '#2ecc71',
                'PANAS': '#e74c3c'
            },
            title="Label Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Row 5: Model Details & Raw Data
    st.subheader("🔧 Model Details")
    
    with st.expander("View Model Information"):
        for model_name, model in models.items():
            st.write(f"**{model_name}**")
            st.write(f"Type: {type(model).__name__}")
            
            if hasattr(model, 'get_params'):
                params = model.get_params()
                st.write(f"Parameters: {len(params)}")
                
                with st.expander("View Parameters"):
                    for key, value in list(params.items())[:10]:  # Show first 10
                        st.write(f"- {key}: {value}")
            
            st.write("---")
    
    with st.expander("View Raw Prediction Data"):
        pred_data = []
        for model_name, pred in predictions.items():
            row = {
                'Model': model_name,
                'Prediction': pred['label'],
                'Confidence': f"{pred['confidence']:.1%}",
                'DINGIN Prob': f"{pred.get('probabilities', {}).get('DINGIN', 0):.1%}",
                'NORMAL Prob': f"{pred.get('probabilities', {}).get('NORMAL', 0):.1%}",
                'PANAS Prob': f"{pred.get('probabilities', {}).get('PANAS', 0):.1%}"
            }
            pred_data.append(row)
        
        st.dataframe(pd.DataFrame(pred_data), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🎯 System Information
    - **Models**: Decision Tree, K-Nearest Neighbors, Logistic Regression
    - **Input Features**: Temperature, Humidity, Hour, Minute
    - **Output Labels**: DINGIN (<22°C), NORMAL (22-25°C), PANAS (>25°C)
    - **Update Frequency**: Real-time (when sensor data received)
    """)
    
    # Auto-refresh
    st.markdown("---")
    if st.button("🔄 Refresh Predictions"):
        st.rerun()

# ==================== REAL-TIME SIMULATION ====================
def real_time_simulation():
    """Simulasi data real-time"""
    st.sidebar.subheader("🔄 Real-time Simulation")
    
    if st.sidebar.button("🎬 Start Simulation"):
        placeholder = st.empty()
        
        for i in range(10):  # 10 updates
            with placeholder.container():
                # Generate random data
                temp = np.random.uniform(18, 32)
                hum = np.random.uniform(40, 80)
                hour = datetime.now().hour
                minute = datetime.now().minute
                
                st.write(f"Simulation #{i+1}")
                st.write(f"Temperature: {temp:.1f}°C")
                st.write(f"Humidity: {hum:.1f}%")
                
                # Load models
                models, scaler = load_models()
                if models and scaler:
                    predictions = predict_temperature(models, scaler, temp, hum, hour, minute)
                    
                    for model_name, pred in predictions.items():
                        st.write(f"{model_name}: {pred['label']} ({pred['confidence']:.1%})")
                
            time.sleep(1)  # Wait 1 second between updates
        
        st.success("✅ Simulation complete!")

if __name__ == "__main__":
    main()
    # Uncomment untuk simulasi real-time
    # real_time_simulation()