import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add src to path
sys.path.append('src')

from src.data_generator import SalesDataGenerator
from src.embedding_trainer import SalesEmbeddingTrainer
from src.evaluator import EmbeddingEvaluator
from src.langchain_pipeline import SalesPipeline

# Page configuration
st.set_page_config(
    page_title="Sales Conversion Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎯 Sales Conversion Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Data Generation", "Model Training", "Evaluation", "Prediction", "Analytics"]
    )
    
    # Initialize session state
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    
    # Page routing
    if page == "Overview":
        show_overview()
    elif page == "Data Generation":
        show_data_generation()
    elif page == "Model Training":
        show_model_training()
    elif page == "Evaluation":
        show_evaluation()
    elif page == "Prediction":
        show_prediction()
    elif page == "Analytics":
        show_analytics()

def show_overview():
    """Overview page with system description and status"""
    st.header("🔍 System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What is this system?")
        st.write("""
        This AI system fine-tunes embeddings specifically for sales conversations 
        to improve conversion prediction accuracy. It helps sales teams:
        
        - **Predict conversion likelihood** from call transcripts
        - **Prioritize prospects** based on buying signals
        - **Compare performance** of fine-tuned vs generic embeddings
        - **Get actionable recommendations** for each prospect
        """)
        
        st.subheader("Key Features")
        st.write("""
        - 🎯 **Domain-specific fine-tuning** for sales conversations
        - 🔄 **Contrastive learning** to distinguish conversion patterns
        - 🔗 **LangChain integration** for pipeline orchestration
        - 📊 **Performance comparison** with generic embeddings
        - 🎨 **Interactive UI** for easy interaction
        """)
    
    with col2:
        st.subheader("System Status")
        
        # Check data status
        train_exists = os.path.exists('data/train_conversations.csv')
        test_exists = os.path.exists('data/test_conversations.csv')
        model_exists = os.path.exists('models/best_sales_embedding_model.pt')
        
        status_data = {
            "Component": ["Training Data", "Test Data", "Fine-tuned Model"],
            "Status": [
                "✅ Ready" if train_exists else "❌ Not Available",
                "✅ Ready" if test_exists else "❌ Not Available", 
                "✅ Ready" if model_exists else "❌ Not Available"
            ]
        }
        
        status_df = pd.DataFrame(status_data)
        st.table(status_df)
        
        if train_exists and test_exists:
            train_df = pd.read_csv('data/train_conversations.csv')
            test_df = pd.read_csv('data/test_conversations.csv')
            
            st.subheader("Dataset Info")
            st.write(f"**Training samples:** {len(train_df)}")
            st.write(f"**Test samples:** {len(test_df)}")
            st.write(f"**Training conversion rate:** {train_df['conversion_label'].mean():.2%}")
            st.write(f"**Test conversion rate:** {test_df['conversion_label'].mean():.2%}")
    
    # Quick start guide
    st.header("🚀 Quick Start Guide")
    
    steps = [
        ("1. Generate Data", "Create synthetic sales conversation data", "data_generation"),
        ("2. Train Model", "Fine-tune embeddings on sales data", "model_training"),
        ("3. Evaluate", "Compare performance with generic embeddings", "evaluation"),
        ("4. Predict", "Make predictions on new conversations", "prediction")
    ]
    
    cols = st.columns(4)
    for i, (title, desc, _) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

def show_data_generation():
    """Data generation page"""
    st.header("📊 Data Generation")
    
    st.write("""
    Generate synthetic sales conversation data with realistic patterns for training and testing.
    The generator creates conversations with distinct patterns for high and low conversion scenarios.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generation Parameters")
        
        num_train = st.slider("Training Samples", 100, 2000, 800, 100)
        num_test = st.slider("Test Samples", 50, 500, 200, 50)
        
        if st.button("Generate Dataset", type="primary"):
            with st.spinner("Generating synthetic data..."):
                generator = SalesDataGenerator()
                
                # Generate training data
                train_df = generator.generate_dataset(num_samples=num_train)
                train_df = generator.add_metadata(train_df)
                
                # Generate test data
                test_df = generator.generate_dataset(num_samples=num_test)
                test_df = generator.add_metadata(test_df)
                
                # Save datasets
                os.makedirs('data', exist_ok=True)
                train_df.to_csv('data/train_conversations.csv', index=False)
                test_df.to_csv('data/test_conversations.csv', index=False)
                
                st.session_state.data_generated = True
                
                st.success(f"Generated {len(train_df)} training and {len(test_df)} test samples!")
    
    with col2:
        st.subheader("Data Preview")
        
        if os.path.exists('data/train_conversations.csv'):
            train_df = pd.read_csv('data/train_conversations.csv')
            
            # Show statistics
            st.metric("Training Samples", len(train_df))
            st.metric("Conversion Rate", f"{train_df['conversion_label'].mean():.2%}")
            
            # Show distribution
            fig = px.bar(
                x=['Low Conversion', 'High Conversion'],
                y=[
                    len(train_df[train_df['conversion_label'] == 0]),
                    len(train_df[train_df['conversion_label'] == 1])
                ],
                title="Label Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Show sample conversations
    if os.path.exists('data/train_conversations.csv'):
        st.subheader("Sample Conversations")
        
        train_df = pd.read_csv('data/train_conversations.csv')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**High Conversion Example:**")
            high_conv = train_df[train_df['conversion_label'] == 1].iloc[0]
            st.write(f"*{high_conv['conversation']}*")
            
        with col2:
            st.write("**Low Conversion Example:**")
            low_conv = train_df[train_df['conversion_label'] == 0].iloc[0]
            st.write(f"*{low_conv['conversation']}*")

def show_model_training():
    """Model training page"""
    st.header("🎓 Model Training")
    
    if not os.path.exists('data/train_conversations.csv'):
        st.warning("Please generate data first before training the model.")
        return
    
    st.write("""
    Fine-tune embeddings using contrastive learning to distinguish between 
    high and low conversion conversation patterns.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Training Parameters")
        
        epochs = st.slider("Epochs", 1, 10, 3)
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1)
        
        if st.button("Start Training", type="primary"):
            with st.spinner("Training model... This may take several minutes."):
                try:
                    # Load data
                    train_df = pd.read_csv('data/train_conversations.csv')
                    test_df = pd.read_csv('data/test_conversations.csv')
                    
                    # Initialize trainer
                    trainer = SalesEmbeddingTrainer()
                    
                    # Train model
                    history = trainer.train(train_df, test_df, epochs=epochs, batch_size=batch_size)
                    
                    st.session_state.model_trained = True
                    st.success("Model training completed!")
                    
                    # Show training history
                    st.subheader("Training History")
                    history_df = pd.DataFrame(history)
                    
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Training Loss', 'Validation Loss', 'Accuracy', 'F1 Score')
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=history_df['epoch'], y=history_df['train_loss'], name='Train Loss'),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=history_df['epoch'], y=history_df['eval_loss'], name='Eval Loss'),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Scatter(x=history_df['epoch'], y=history_df['eval_accuracy'], name='Accuracy'),
                        row=2, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=history_df['epoch'], y=history_df['eval_f1'], name='F1 Score'),
                        row=2, col=2
                    )
                    
                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
    with col2:
        st.subheader("Training Status")
        
        if os.path.exists('models/best_sales_embedding_model.pt'):
            st.success("✅ Model trained and saved")
            
            # Show model info
            st.write("**Model Details:**")
            st.write("- Base Model: all-MiniLM-L6-v2")
            st.write("- Training: Contrastive Learning")
            st.write("- Task: Binary Classification")
            
        else:
            st.info("No trained model found")

def show_evaluation():
    """Model evaluation page"""
    st.header("📈 Model Evaluation")
    
    if not os.path.exists('models/best_sales_embedding_model.pt'):
        st.warning("Please train the model first before evaluation.")
        return
    
    st.write("""
    Compare the performance of fine-tuned embeddings against generic embeddings
    on the sales conversation dataset.
    """)
    
    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Running evaluation..."):
            try:
                # Load data
                train_df = pd.read_csv('data/train_conversations.csv')
                test_df = pd.read_csv('data/test_conversations.csv')
                
                # Initialize evaluator
                evaluator = EmbeddingEvaluator()
                evaluator.load_fine_tuned_model('models/best_sales_embedding_model.pt')
                
                # Run evaluation
                results = evaluator.evaluate_embeddings(train_df, test_df)
                
                # Display results
                st.subheader("Performance Comparison")
                
                # Create comparison table
                comparison_data = []
                for embedding_type in results:
                    for classifier_type in results[embedding_type]:
                        metrics = results[embedding_type][classifier_type]
                        comparison_data.append({
                            'Embedding Type': embedding_type.replace('_', ' ').title(),
                            'Classifier': classifier_type.replace('_', ' ').title(),
                            'Accuracy': f"{metrics['accuracy']:.4f}",
                            'Precision': f"{metrics['precision']:.4f}",
                            'Recall': f"{metrics['recall']:.4f}",
                            'F1 Score': f"{metrics['f1']:.4f}"
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualization
                st.subheader("Performance Visualization")
                
                # Prepare data for plotting
                metrics_data = []
                for embedding_type in results:
                    for classifier_type in results[embedding_type]:
                        metrics = results[embedding_type][classifier_type]
                        for metric_name, value in metrics.items():
                            metrics_data.append({
                                'Embedding': embedding_type.replace('_', ' ').title(),
                                'Classifier': classifier_type.replace('_', ' ').title(),
                                'Metric': metric_name.title(),
                                'Value': value
                            })
                
                metrics_df = pd.DataFrame(metrics_data)
                
                fig = px.bar(
                    metrics_df,
                    x='Metric',
                    y='Value',
                    color='Embedding',
                    facet_col='Classifier',
                    title='Performance Comparison: Fine-tuned vs Generic Embeddings'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate report
                report = evaluator.generate_report()
                st.subheader("Detailed Report")
                st.markdown(report)
                
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")

def show_prediction():
    """Prediction page"""
    st.header("🔮 Conversion Prediction")
    
    if not os.path.exists('models/best_sales_embedding_model.pt'):
        st.warning("Please train the model first before making predictions.")
        return
    
    # Initialize pipeline
    if st.session_state.pipeline is None:
        conversations_df = None
        if os.path.exists('data/train_conversations.csv'):
            conversations_df = pd.read_csv('data/train_conversations.csv')
        
        st.session_state.pipeline = SalesPipeline(
            model_path='models/best_sales_embedding_model.pt',
            conversations_df=conversations_df
        )
    
    st.write("Enter a sales conversation to predict conversion likelihood:")
    
    # Sample conversations
    sample_conversations = [
        "I'm very interested in this solution and would like to schedule a demo for next week. Can you send me pricing information?",
        "I need to think about it and discuss with my team before making any decisions. The timing might not be right.",
        "This looks perfect for our needs. When can we get started with implementation? I'm ready to move forward.",
        "We're happy with our current solution and don't see a need to change right now."
    ]
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Type conversation", "Select sample"])
    
    if input_method == "Type conversation":
        conversation = st.text_area("Enter conversation:", height=150)
    else:
        conversation = st.selectbox("Select sample conversation:", sample_conversations)
    
    if st.button("Predict Conversion", type="primary") and conversation:
        with st.spinner("Analyzing conversation..."):
            try:
                # Get prediction
                result = st.session_state.pipeline.predict_conversion(conversation)
                
                # Display prediction
                st.subheader("Prediction Result")
                st.text(result['prediction'])
                
                # Get recommendations
                recommendations = st.session_state.pipeline.get_recommendations(conversation)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Recommendations")
                    st.write(f"**Priority:** {recommendations['priority'].title()}")
                    st.write("**Suggested Actions:**")
                    for action in recommendations['actions']:
                        st.write(f"• {action}")
                
                with col2:
                    st.subheader("Similar Conversations")
                    if result['similar_conversations']:
                        for i, similar in enumerate(result['similar_conversations'][:3]):
                            with st.expander(f"Similar #{i+1} - {'✅ Converted' if similar['conversion_label'] else '❌ No Conversion'}"):
                                st.write(similar['conversation'])
                                st.write(f"**Industry:** {similar['industry']}")
                                st.write(f"**Company Size:** {similar['company_size']}")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

def show_analytics():
    """Analytics page"""
    st.header("📊 Analytics Dashboard")
    
    if not os.path.exists('data/train_conversations.csv'):
        st.warning("Please generate data first.")
        return
    
    # Load data
    train_df = pd.read_csv('data/train_conversations.csv')
    
    # Dataset overview
    st.subheader("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Conversations", len(train_df))
    with col2:
        st.metric("Conversion Rate", f"{train_df['conversion_label'].mean():.2%}")
    with col3:
        st.metric("Avg Call Duration", f"{train_df['call_duration_minutes'].mean():.1f} min")
    with col4:
        st.metric("Unique Sales Reps", train_df['sales_rep'].nunique())
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Conversion by industry
        industry_conv = train_df.groupby('industry')['conversion_label'].mean().reset_index()
        fig = px.bar(
            industry_conv,
            x='industry',
            y='conversion_label',
            title='Conversion Rate by Industry'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Call duration distribution
        fig = px.histogram(
            train_df,
            x='call_duration_minutes',
            color='conversion_label',
            title='Call Duration Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis
    st.subheader("Detailed Analysis")
    
    # Conversion by company size
    company_analysis = train_df.groupby(['company_size', 'conversion_label']).size().unstack(fill_value=0)
    company_analysis['conversion_rate'] = company_analysis[1] / (company_analysis[0] + company_analysis[1])
    
    fig = px.bar(
        x=company_analysis.index,
        y=company_analysis['conversion_rate'],
        title='Conversion Rate by Company Size'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Sales rep performance
    rep_performance = train_df.groupby('sales_rep').agg({
        'conversion_label': ['count', 'mean'],
        'call_duration_minutes': 'mean'
    }).round(3)
    
    rep_performance.columns = ['Total Calls', 'Conversion Rate', 'Avg Duration']
    st.subheader("Sales Rep Performance")
    st.dataframe(rep_performance, use_container_width=True)

if __name__ == "__main__":
    main() 