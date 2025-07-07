from langchain.schema import BaseRetriever, Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.runnable import Runnable
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from .embedding_trainer import SalesEmbeddingTrainer
from .evaluator import EmbeddingEvaluator
import os

class SalesEmbeddings(Embeddings):
    """Custom LangChain embeddings wrapper for sales-specific embeddings"""
    
    def __init__(self, model_path: str = None):
        self.trainer = SalesEmbeddingTrainer()
        if model_path and os.path.exists(model_path):
            self.trainer.load_model(model_path)
            self.is_fine_tuned = True
        else:
            self.is_fine_tuned = False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if self.is_fine_tuned:
            embeddings = self.trainer.get_embeddings(texts)
        else:
            # Use generic embeddings if fine-tuned model not available
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            embeddings = model.encode(texts)
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]

class SalesConversionPredictor(LLM):
    """Custom LLM for sales conversion prediction"""
    
    def __init__(self, model_path: str = None):
        super().__init__()
        self.trainer = SalesEmbeddingTrainer()
        if model_path and os.path.exists(model_path):
            self.trainer.load_model(model_path)
            self.is_fine_tuned = True
        else:
            self.is_fine_tuned = False
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Make prediction based on the prompt"""
        if self.is_fine_tuned:
            predictions = self.trainer.predict_conversion([prompt])
            prediction = predictions[0]
            
            probability = prediction['conversion_probability']
            predicted_label = prediction['predicted_label']
            
            # Format response
            if predicted_label == 1:
                result = f"HIGH CONVERSION LIKELIHOOD ({probability:.2%})"
                recommendation = "PRIORITIZE: This prospect shows strong buying signals."
            else:
                result = f"LOW CONVERSION LIKELIHOOD ({probability:.2%})"
                recommendation = "NURTURE: This prospect needs more engagement."
            
            return f"""
CONVERSION PREDICTION: {result}

PROBABILITY SCORE: {probability:.2%}

RECOMMENDATION: {recommendation}

ANALYSIS: Based on the conversation patterns, this prospect demonstrates {'positive' if predicted_label == 1 else 'neutral/negative'} indicators for conversion.
"""
        else:
            return "Fine-tuned model not available. Please train the model first."
    
    @property
    def _llm_type(self) -> str:
        return "sales_conversion_predictor"

class SalesConversationRetriever(BaseRetriever):
    """Custom retriever for finding similar sales conversations"""
    
    def __init__(self, conversations_df: pd.DataFrame, embeddings: SalesEmbeddings, k: int = 5):
        super().__init__()
        self.conversations_df = conversations_df
        self.embeddings = embeddings
        self.k = k
        
        # Create vector store
        documents = [
            Document(
                page_content=row['conversation'],
                metadata={
                    'conversion_label': row['conversion_label'],
                    'conversation_id': row['conversation_id'],
                    'industry': row.get('industry', 'Unknown'),
                    'company_size': row.get('company_size', 'Unknown')
                }
            )
            for _, row in conversations_df.iterrows()
        ]
        
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """Retrieve relevant conversations"""
        return self.vectorstore.similarity_search(query, k=self.k)

class SalesPipeline:
    """Main pipeline for sales conversion prediction and analysis"""
    
    def __init__(self, model_path: str = None, conversations_df: pd.DataFrame = None):
        self.model_path = model_path
        self.conversations_df = conversations_df
        
        # Initialize components
        self.embeddings = SalesEmbeddings(model_path)
        self.predictor = SalesConversionPredictor(model_path)
        
        if conversations_df is not None:
            self.retriever = SalesConversationRetriever(conversations_df, self.embeddings)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.predictor,
                chain_type="stuff",
                retriever=self.retriever
            )
    
    def predict_conversion(self, conversation: str) -> Dict[str, Any]:
        """Predict conversion probability for a conversation"""
        # Get prediction
        prediction_result = self.predictor(conversation)
        
        # Get similar conversations if retriever available
        similar_conversations = []
        if hasattr(self, 'retriever'):
            similar_docs = self.retriever.get_relevant_documents(conversation)
            similar_conversations = [
                {
                    'conversation': doc.page_content,
                    'conversion_label': doc.metadata['conversion_label'],
                    'industry': doc.metadata['industry'],
                    'company_size': doc.metadata['company_size']
                }
                for doc in similar_docs
            ]
        
        return {
            'prediction': prediction_result,
            'similar_conversations': similar_conversations,
            'conversation': conversation
        }
    
    def batch_predict(self, conversations: List[str]) -> List[Dict[str, Any]]:
        """Predict conversion for multiple conversations"""
        results = []
        for conversation in conversations:
            result = self.predict_conversion(conversation)
            results.append(result)
        return results
    
    def analyze_conversation_patterns(self, conversations: List[str]) -> Dict[str, Any]:
        """Analyze patterns in a set of conversations"""
        if not self.embeddings.is_fine_tuned:
            return {"error": "Fine-tuned model required for pattern analysis"}
        
        # Get embeddings for all conversations
        embeddings = np.array(self.embeddings.embed_documents(conversations))
        
        # Get predictions
        predictions = []
        for conv in conversations:
            pred_result = self.predictor(conv)
            predictions.append(pred_result)
        
        # Calculate similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find conversation clusters
        from sklearn.cluster import KMeans
        n_clusters = min(3, len(conversations))  # Max 3 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_conversations = [conversations[j] for j in range(len(conversations)) if clusters[j] == i]
            cluster_predictions = [predictions[j] for j in range(len(predictions)) if clusters[j] == i]
            
            cluster_analysis[f'cluster_{i}'] = {
                'size': len(cluster_conversations),
                'conversations': cluster_conversations,
                'predictions': cluster_predictions
            }
        
        return {
            'total_conversations': len(conversations),
            'similarity_matrix': similarity_matrix.tolist(),
            'clusters': cluster_analysis,
            'predictions': predictions
        }
    
    def get_recommendations(self, conversation: str) -> Dict[str, Any]:
        """Get actionable recommendations based on conversation analysis"""
        prediction_result = self.predict_conversion(conversation)
        
        # Extract probability from prediction result
        prediction_text = prediction_result['prediction']
        
        # Parse probability (basic extraction)
        if 'HIGH CONVERSION' in prediction_text:
            conversion_likelihood = 'high'
            priority = 'immediate'
        else:
            conversion_likelihood = 'low'
            priority = 'nurture'
        
        # Generate recommendations
        recommendations = {
            'conversion_likelihood': conversion_likelihood,
            'priority': priority,
            'actions': self._generate_actions(conversion_likelihood),
            'similar_conversations': prediction_result['similar_conversations']
        }
        
        return recommendations
    
    def _generate_actions(self, likelihood: str) -> List[str]:
        """Generate action recommendations based on likelihood"""
        if likelihood == 'high':
            return [
                "Schedule follow-up meeting within 24-48 hours",
                "Prepare detailed proposal with pricing",
                "Identify decision makers and stakeholders",
                "Create implementation timeline",
                "Prepare contract and legal documents"
            ]
        else:
            return [
                "Send educational content about product benefits",
                "Schedule demo or product walkthrough",
                "Address specific concerns mentioned in conversation",
                "Provide case studies from similar companies",
                "Set up regular check-ins to maintain engagement"
            ]
    
    def train_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                   epochs: int = 3, batch_size: int = 8) -> Dict[str, Any]:
        """Train the fine-tuned model"""
        trainer = SalesEmbeddingTrainer()
        
        # Train model
        history = trainer.train(train_df, test_df, epochs=epochs, batch_size=batch_size)
        
        # Update pipeline with trained model
        self.embeddings = SalesEmbeddings('models/best_sales_embedding_model.pt')
        self.predictor = SalesConversionPredictor('models/best_sales_embedding_model.pt')
        
        if self.conversations_df is not None:
            self.retriever = SalesConversationRetriever(self.conversations_df, self.embeddings)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.predictor,
                chain_type="stuff",
                retriever=self.retriever
            )
        
        return {
            'training_history': history,
            'model_path': 'models/best_sales_embedding_model.pt'
        }
    
    def evaluate_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance"""
        evaluator = EmbeddingEvaluator()
        
        if os.path.exists('models/best_sales_embedding_model.pt'):
            evaluator.load_fine_tuned_model('models/best_sales_embedding_model.pt')
        
        # Run evaluation
        results = evaluator.evaluate_embeddings(train_df, test_df)
        report = evaluator.generate_report()
        
        return {
            'results': results,
            'report': report,
            'evaluator': evaluator
        }

def main():
    """Example usage of the sales pipeline"""
    # Load data
    if os.path.exists('data/train_conversations.csv'):
        train_df = pd.read_csv('data/train_conversations.csv')
        test_df = pd.read_csv('data/test_conversations.csv')
        
        # Initialize pipeline
        pipeline = SalesPipeline(
            model_path='models/best_sales_embedding_model.pt' if os.path.exists('models/best_sales_embedding_model.pt') else None,
            conversations_df=train_df
        )
        
        # Example conversation
        sample_conversation = "I'm very interested in this solution and would like to schedule a demo for next week. Can you send me pricing information?"
        
        # Get prediction
        result = pipeline.predict_conversion(sample_conversation)
        print("Prediction Result:")
        print(result['prediction'])
        
        # Get recommendations
        recommendations = pipeline.get_recommendations(sample_conversation)
        print("\nRecommendations:")
        print(f"Priority: {recommendations['priority']}")
        print(f"Actions: {recommendations['actions']}")
        
    else:
        print("No training data found. Please generate data first.")

if __name__ == "__main__":
    main() 