import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import plotly.express as px
import plotly.graph_objects as go
import time
import math

# Enable CPU optimizations
try:
    tf.config.optimizer.set_jit(True)
    physical_devices = tf.config.list_physical_devices('CPU')
    tf.config.threading.set_inter_op_parallelism_threads(len(physical_devices))
except:
    pass

# Page config
st.set_page_config(
    page_title="Classification Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .main-title {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem 0;
    }
    .section-title {
        color: #2ecc71;
        font-size: 1.8rem;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }
    </style>
""", unsafe_allow_html=True)

def calculate_metrics(y_true, y_pred):
    """Calculate metrics for multiclass classification with debugging"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'gmean': math.sqrt(precision * recall)
    }


def plot_metrics(metrics, title):
    """Plot performance metrics with improved visualization"""
    metric_values = {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1 Score': metrics['f1_score'],
        'G-Mean': metrics['gmean']
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(metric_values.keys()),
            y=list(metric_values.values()),
            text=[f'{v:.2f}%' for v in metric_values.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 100],
        width=500,
        height=500,
        showlegend=False
    )
    
    return fig

def plot_confusion_matrix(cm, title):
    """Plot confusion matrix with proper multiclass handling"""
    classes = range(cm.shape[0])
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f'Predicted {i}' for i in classes],
        y=[f'Actual {i}' for i in classes],
        text=cm,
        texttemplate="%{text}",
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=500,
        height=500
    )
    return fig

def run_neural_network(X, y, neurons, batch_size, epochs):
    """Run neural network training with correct class handling"""
    # Determine number of classes correctly
    y_min = np.min(y)
    y_max = np.max(y)
    num_classes = y_max - y_min + 1
    
    
    # Convert target to categorical, adjusting for class starting number
    y_categorical = tf.keras.utils.to_categorical(y - y_min, num_classes=num_classes)
    
    # Print shapes for debugging
    st.write(f"Input shape: {X.shape}")
    st.write(f"Target shape: {y_categorical.shape}")
    
    # Ensure proper data types
    X = X.astype('float32')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
    # Create model with correct number of output neurons
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(neurons, activation='relu', kernel_initializer='he_normal'),
        Dense(neurons//2, activation='relu', kernel_initializer='he_normal'),
        Dense(num_classes, activation='softmax')  # Matches number of classes
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_place = st.empty()
    
    train_metrics = []
    val_metrics = []
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    for epoch in range(epochs):
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=1,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[early_stopping]
        )
        
        train_acc = history.history['accuracy'][0]
        val_acc = history.history['val_accuracy'][0]
        
        train_metrics.append(train_acc)
        val_metrics.append(val_acc)
        
        # Update progress
        progress_bar.progress((epoch + 1) / epochs)
        status_text.markdown(f"""
            **Epoch {epoch + 1}/{epochs}**
            - Training Accuracy: {train_acc:.4f}
            - Validation Accuracy: {val_acc:.4f}
        """)
        
        # Update training plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train_metrics, name='Training', mode='lines+markers'))
        fig.add_trace(go.Scatter(y=val_metrics, name='Validation', mode='lines+markers'))
        fig.update_layout(
            title='Training Progress',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            yaxis_range=[0, 1]
        )
        chart_place.plotly_chart(fig)
        
        time.sleep(0.1)
    
    # Final predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1) + y_min  # Add back the minimum class value
    y_test_classes = np.argmax(y_test, axis=1) + y_min  # Add back the minimum class value
    
    metrics = calculate_metrics(y_test_classes, y_pred)
    return metrics, model

def run_svm_analysis(X, y):
    """Run SVM classification with proper class handling and debugging"""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model with progress indicator
    with st.spinner('Training SVM...'):
        svm = SVC(
            kernel='rbf', 
            gamma='scale', 
            C=1.0, 
            decision_function_shape='ovr',
            random_state=42
        )
        svm.fit(X_train, y_train)
    
    # Predictions with debug info
    y_pred = svm.predict(X_test)
    
    return calculate_metrics(y_test, y_pred)

def run_knn_analysis(X, y, n_neighbors):
    """Run KNN classification with proper class handling and debugging"""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model with progress indicator
    with st.spinner('Training KNN...'):
        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='uniform',
            algorithm='auto'
        )
        knn.fit(X_train, y_train)
    
    # Predictions with debug info
    y_pred = knn.predict(X_test)
    
    return calculate_metrics(y_test, y_pred)

def run_naive_bayes_analysis(X, y):
    """Run Naive Bayes classification with proper class handling and debugging"""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model with progress indicator
    with st.spinner('Training Naive Bayes...'):
        nb = GaussianNB()
        nb.fit(X_train, y_train)
    
    # Predictions with debug info
    y_pred = nb.predict(X_test)
    
    return calculate_metrics(y_test, y_pred)



def main():
    st.markdown('<p class="title">🤖 Machine Learning Classification Analysis</p>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown('### 🧭 Navigation')
    page = st.sidebar.radio('Select Page', [
        '📊 Data Overview',
        '🧠 Neural Network',
        '🎯 SVM',
        '🔍 KNN',
        '📈 Naive Bayes',
        '🏆 Model Comparison'
    ], label_visibility="collapsed")
    
    try:
        # Load and preprocess data
        data = pd.read_excel("classification data.xlsx")
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if page == '📊 Data Overview':
            st.markdown('<p class="subtitle">📊 Dataset Overview</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Data Sample")
                st.dataframe(data.head())
            
            with col2:
                st.write("Dataset Statistics")
                st.dataframe(data.describe())
            
            st.write("Feature Correlations")
            correlation_matrix = data.corr()
            fig = px.imshow(correlation_matrix, 
                          aspect='auto',
                          title='Feature Correlation Heatmap')
            st.plotly_chart(fig)
        
        elif page == '🧠 Neural Network':
            st.markdown('<p class="subtitle">🧠 Neural Network Analysis</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                neurons = st.selectbox('Hidden Neurons:', [10, 20, 30])
            with col2:
                batch_size = st.selectbox('Batch Size:', [15, 30, 45])
            with col3:
                epochs = st.slider('Number of Epochs:', 5, 50, 20)
            
            if st.button('🚀 Train Neural Network'):
                metrics, model = run_neural_network(X_scaled, y, neurons, batch_size, epochs)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_confusion_matrix(metrics['confusion_matrix'], 
                                                        'Neural Network Confusion Matrix'))
                with col2:
                    st.plotly_chart(plot_metrics(metrics, 'Neural Network Metrics'))
        
        elif page == '🎯 SVM':
            st.markdown('<p class="subtitle">🎯 Support Vector Machine Analysis</p>', 
                       unsafe_allow_html=True)
            
            if st.button('🚀 Train SVM'):
                with st.spinner('Training SVM...'):
                    metrics = run_svm_analysis(X_scaled, y)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(plot_confusion_matrix(metrics['confusion_matrix'], 
                                                            'SVM Confusion Matrix'))
                    with col2:
                        st.plotly_chart(plot_metrics(metrics, 'SVM Metrics'))
        
        elif page == '🔍 KNN':
            st.markdown('<p class="subtitle">🔍 K-Nearest Neighbors Analysis</p>', 
                       unsafe_allow_html=True)
            
            n_neighbors = st.slider('Number of Neighbors (K):', 1, 50, 25)
            
            if st.button('🚀 Train KNN'):
                with st.spinner('Training KNN...'):
                    metrics = run_knn_analysis(X_scaled, y, n_neighbors)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(plot_confusion_matrix(metrics['confusion_matrix'], 
                                                            'KNN Confusion Matrix'))
                    with col2:
                        st.plotly_chart(plot_metrics(metrics, 'KNN Metrics'))
        
        elif page == '📈 Naive Bayes':
            st.markdown('<p class="subtitle">📈 Naive Bayes Analysis</p>', 
                       unsafe_allow_html=True)
            
            if st.button('🚀 Train Naive Bayes'):
                with st.spinner('Training Naive Bayes...'):
                    metrics = run_naive_bayes_analysis(X_scaled, y)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(plot_confusion_matrix(metrics['confusion_matrix'], 
                                                            'Naive Bayes Confusion Matrix'))
                    with col2:
                        st.plotly_chart(plot_metrics(metrics, 'Naive Bayes Metrics'))
        
        else:  # Model Comparison
            st.markdown('<p class="subtitle">🏆 Model Performance Comparison</p>', unsafe_allow_html=True)
            
            # Add configuration options
            st.sidebar.markdown("### Model Configuration")
            compare_config = st.sidebar.expander("Compare Configuration", expanded=False)
            with compare_config:
                nn_epochs = st.number_input("Neural Network Epochs", min_value=5, max_value=100, value=10)
                nn_neurons = st.number_input("Neural Network Neurons", min_value=10, max_value=100, value=20)
                knn_neighbors = st.number_input("KNN Neighbors", min_value=1, max_value=50, value=25)
            
            if st.button('🚀 Run Model Comparison'):
                try:
                    # Container for results
                    results = {}
                    
                    # Progress container
                    progress_container = st.empty()
                    metrics_container = st.empty()
                    
                    # Neural Network
                    progress_container.markdown("🧠 Training Neural Network...")
                    nn_metrics, _ = run_neural_network(X_scaled, y, nn_neurons, 32, nn_epochs)
                    results['Neural Network'] = nn_metrics
                    
                    # SVM
                    progress_container.markdown("🎯 Training SVM...")
                    results['SVM'] = run_svm_analysis(X_scaled, y)
                    
                    # KNN
                    progress_container.markdown("🔍 Training KNN...")
                    results['KNN'] = run_knn_analysis(X_scaled, y, knn_neighbors)
                    
                    # Naive Bayes
                    progress_container.markdown("📈 Training Naive Bayes...")
                    results['Naive Bayes'] = run_naive_bayes_analysis(X_scaled, y)
                    
                    progress_container.empty()
                    
                    # Create comparison data
                    comparison_data = []
                    for model_name, metrics in results.items():
                        comparison_data.append({
                            'Model': model_name,
                            'Accuracy': metrics['accuracy'],
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall'],
                            'F1 Score': metrics['f1_score'],
                            'G-Mean': metrics['gmean']
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Display detailed metrics table
                    st.markdown("### Detailed Metrics")
                    st.dataframe(comparison_df.style.format({
                        'Accuracy': '{:.2f}%',
                        'Precision': '{:.2f}%',
                        'Recall': '{:.2f}%',
                        'F1 Score': '{:.2f}%',
                        'G-Mean': '{:.2f}%'
                    }))
                    
                    # Create performance comparison plot
                    fig = px.bar(
                        comparison_df.melt(
                            id_vars=['Model'],
                            var_name='Metric',
                            value_name='Score'
                        ),
                        x='Model',
                        y='Score',
                        color='Metric',
                        title='Model Performance Comparison',
                        labels={'Score': 'Percentage (%)'},
                        height=500
                    )
                    
                    fig.update_layout(
                        yaxis_range=[0, 100],
                        xaxis_title="Model",
                        yaxis_title="Score (%)",
                        legend_title="Metric",
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display best model for each metric
                    st.markdown("### 🏆 Best Models")
                    cols = st.columns(3)
                    
                    with cols[0]:
                        best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
                        st.info(f"""
                        **Best Accuracy**
                        Model: {best_accuracy['Model']}
                        Score: {best_accuracy['Accuracy']:.2f}%
                        """)
                    
                    with cols[1]:
                        best_f1 = comparison_df.loc[comparison_df['F1 Score'].idxmax()]
                        st.info(f"""
                        **Best F1 Score**
                        Model: {best_f1['Model']}
                        Score: {best_f1['F1 Score']:.2f}%
                        """)
                    
                    with cols[2]:
                        best_gmean = comparison_df.loc[comparison_df['G-Mean'].idxmax()]
                        st.info(f"""
                        **Best G-Mean**
                        Model: {best_gmean['Model']}
                        Score: {best_gmean['G-Mean']:.2f}%
                        """)
                    
                    # Add model recommendations
                    st.markdown("### 💡 Model Recommendations")
                    accuracies = comparison_df['Accuracy'].values
                    accuracy_spread = np.std(accuracies)
                    
                    if accuracy_spread < 5:
                        st.info("All models perform similarly. Consider using the simplest model (Naive Bayes or KNN) for better interpretability.")
                    else:
                        best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
                        st.success(f"""
                        The {best_model['Model']} shows the best overall performance:
                        - Accuracy: {best_model['Accuracy']:.2f}%
                        - F1 Score: {best_model['F1 Score']:.2f}%
                        - G-Mean: {best_model['G-Mean']:.2f}%
                        """)
                
                except Exception as e:
                    st.error(f"An error occurred during model comparison: {str(e)}")
                    st.write("Please check your data and try again.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please ensure the dataset file 'classification data.xlsx' is in the correct location.")

if __name__ == "__main__":
    main()