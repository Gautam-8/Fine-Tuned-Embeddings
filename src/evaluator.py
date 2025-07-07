import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pickle
import os

class EmbeddingEvaluator:
    def __init__(self):
        self.generic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.fine_tuned_model = None
        self.results = {}
        
    def load_fine_tuned_model(self, model_path: str):
        """Load the fine-tuned model"""
        from .embedding_trainer import SalesEmbeddingTrainer
        
        self.fine_tuned_trainer = SalesEmbeddingTrainer()
        self.fine_tuned_trainer.load_model(model_path)
        print("Fine-tuned model loaded successfully")
    
    def get_generic_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from generic model"""
        return self.generic_model.encode(texts)
    
    def get_fine_tuned_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from fine-tuned model"""
        if self.fine_tuned_trainer is None:
            raise ValueError("Fine-tuned model not loaded. Call load_fine_tuned_model() first.")
        
        return self.fine_tuned_trainer.get_embeddings(texts)
    
    def train_classifier(self, embeddings: np.ndarray, labels: np.ndarray, 
                        classifier_type: str = 'logistic') -> object:
        """Train a classifier on embeddings"""
        if classifier_type == 'logistic':
            classifier = LogisticRegression(random_state=42, max_iter=1000)
        elif classifier_type == 'random_forest':
            classifier = RandomForestClassifier(random_state=42, n_estimators=100)
        else:
            raise ValueError("Classifier type must be 'logistic' or 'random_forest'")
        
        classifier.fit(embeddings, labels)
        return classifier
    
    def evaluate_embeddings(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """Comprehensive evaluation of both embedding types"""
        
        # Extract texts and labels
        train_texts = train_df['conversation'].tolist()
        train_labels = train_df['conversion_label'].values
        test_texts = test_df['conversation'].tolist()
        test_labels = test_df['conversion_label'].values
        
        results = {}
        
        # Generic embeddings evaluation
        print("Evaluating generic embeddings...")
        generic_train_embeddings = self.get_generic_embeddings(train_texts)
        generic_test_embeddings = self.get_generic_embeddings(test_texts)
        
        # Train classifiers on generic embeddings
        generic_lr = self.train_classifier(generic_train_embeddings, train_labels, 'logistic')
        generic_rf = self.train_classifier(generic_train_embeddings, train_labels, 'random_forest')
        
        # Evaluate generic embeddings
        generic_lr_pred = generic_lr.predict(generic_test_embeddings)
        generic_rf_pred = generic_rf.predict(generic_test_embeddings)
        
        results['generic'] = {
            'logistic_regression': self._calculate_metrics(test_labels, generic_lr_pred),
            'random_forest': self._calculate_metrics(test_labels, generic_rf_pred)
        }
        
        # Fine-tuned embeddings evaluation
        if self.fine_tuned_trainer is not None:
            print("Evaluating fine-tuned embeddings...")
            fine_tuned_train_embeddings = self.get_fine_tuned_embeddings(train_texts)
            fine_tuned_test_embeddings = self.get_fine_tuned_embeddings(test_texts)
            
            # Train classifiers on fine-tuned embeddings
            fine_tuned_lr = self.train_classifier(fine_tuned_train_embeddings, train_labels, 'logistic')
            fine_tuned_rf = self.train_classifier(fine_tuned_train_embeddings, train_labels, 'random_forest')
            
            # Evaluate fine-tuned embeddings
            fine_tuned_lr_pred = fine_tuned_lr.predict(fine_tuned_test_embeddings)
            fine_tuned_rf_pred = fine_tuned_rf.predict(fine_tuned_test_embeddings)
            
            results['fine_tuned'] = {
                'logistic_regression': self._calculate_metrics(test_labels, fine_tuned_lr_pred),
                'random_forest': self._calculate_metrics(test_labels, fine_tuned_rf_pred)
            }
            
            # Direct fine-tuned model predictions
            direct_predictions = self.fine_tuned_trainer.predict_conversion(test_texts)
            direct_pred_labels = [pred['predicted_label'] for pred in direct_predictions]
            
            results['fine_tuned']['direct_model'] = self._calculate_metrics(test_labels, direct_pred_labels)
        
        self.results = results
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def compare_embeddings_similarity(self, texts: List[str], labels: List[int]) -> Dict:
        """Compare embedding similarity patterns between generic and fine-tuned models"""
        
        # Get embeddings
        generic_embeddings = self.get_generic_embeddings(texts)
        
        results = {'generic': {}}
        
        # Analyze generic embeddings
        results['generic'] = self._analyze_embedding_patterns(generic_embeddings, labels, 'Generic')
        
        # Analyze fine-tuned embeddings if available
        if self.fine_tuned_trainer is not None:
            fine_tuned_embeddings = self.get_fine_tuned_embeddings(texts)
            results['fine_tuned'] = self._analyze_embedding_patterns(fine_tuned_embeddings, labels, 'Fine-tuned')
        
        return results
    
    def _analyze_embedding_patterns(self, embeddings: np.ndarray, labels: List[int], 
                                  model_type: str) -> Dict:
        """Analyze embedding patterns for a specific model type"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Separate by labels
        high_conv_indices = [i for i, label in enumerate(labels) if label == 1]
        low_conv_indices = [i for i, label in enumerate(labels) if label == 0]
        
        # Calculate average similarities
        high_conv_similarities = []
        low_conv_similarities = []
        cross_similarities = []
        
        # High conversion to high conversion
        for i in high_conv_indices:
            for j in high_conv_indices:
                if i != j:
                    high_conv_similarities.append(similarity_matrix[i, j])
        
        # Low conversion to low conversion
        for i in low_conv_indices:
            for j in low_conv_indices:
                if i != j:
                    low_conv_similarities.append(similarity_matrix[i, j])
        
        # Cross similarities (high to low)
        for i in high_conv_indices:
            for j in low_conv_indices:
                cross_similarities.append(similarity_matrix[i, j])
        
        return {
            'avg_high_conv_similarity': np.mean(high_conv_similarities),
            'avg_low_conv_similarity': np.mean(low_conv_similarities),
            'avg_cross_similarity': np.mean(cross_similarities),
            'separation_score': (np.mean(high_conv_similarities) + np.mean(low_conv_similarities)) / 2 - np.mean(cross_similarities)
        }
    
    def plot_comparison(self, save_path: str = None):
        """Plot comparison results"""
        if not self.results:
            raise ValueError("No results to plot. Run evaluate_embeddings() first.")
        
        # Prepare data for plotting
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        models = []
        values = []
        metric_names = []
        
        for embedding_type in self.results:
            for classifier_type in self.results[embedding_type]:
                model_name = f"{embedding_type}_{classifier_type}"
                for metric in metrics:
                    models.append(model_name)
                    values.append(self.results[embedding_type][classifier_type][metric])
                    metric_names.append(metric)
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Model': models,
            'Value': values,
            'Metric': metric_names
        })
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Embedding Comparison Results', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            metric_data = plot_df[plot_df['Metric'] == metric]
            
            sns.barplot(data=metric_data, x='Model', y='Value', ax=ax)
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_ylabel(metric.capitalize())
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        if not self.results:
            raise ValueError("No results to report. Run evaluate_embeddings() first.")
        
        report = "# Embedding Evaluation Report\n\n"
        
        # Performance comparison
        report += "## Performance Comparison\n\n"
        
        for embedding_type in self.results:
            report += f"### {embedding_type.replace('_', ' ').title()} Embeddings\n\n"
            
            for classifier_type in self.results[embedding_type]:
                metrics = self.results[embedding_type][classifier_type]
                report += f"**{classifier_type.replace('_', ' ').title()}:**\n"
                report += f"- Accuracy: {metrics['accuracy']:.4f}\n"
                report += f"- Precision: {metrics['precision']:.4f}\n"
                report += f"- Recall: {metrics['recall']:.4f}\n"
                report += f"- F1 Score: {metrics['f1']:.4f}\n\n"
        
        # Improvement analysis
        if 'fine_tuned' in self.results and 'generic' in self.results:
            report += "## Improvement Analysis\n\n"
            
            for classifier_type in ['logistic_regression', 'random_forest']:
                if classifier_type in self.results['fine_tuned'] and classifier_type in self.results['generic']:
                    fine_tuned_f1 = self.results['fine_tuned'][classifier_type]['f1']
                    generic_f1 = self.results['generic'][classifier_type]['f1']
                    improvement = ((fine_tuned_f1 - generic_f1) / generic_f1) * 100
                    
                    report += f"**{classifier_type.replace('_', ' ').title()}:**\n"
                    report += f"- F1 Score Improvement: {improvement:.2f}%\n"
                    report += f"- Fine-tuned F1: {fine_tuned_f1:.4f}\n"
                    report += f"- Generic F1: {generic_f1:.4f}\n\n"
        
        return report
    
    def save_results(self, path: str):
        """Save evaluation results to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Results saved to {path}")
    
    def load_results(self, path: str):
        """Load evaluation results from file"""
        with open(path, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Results loaded from {path}")

def main():
    """Example usage"""
    # Load data
    train_df = pd.read_csv('data/train_conversations.csv')
    test_df = pd.read_csv('data/test_conversations.csv')
    
    # Initialize evaluator
    evaluator = EmbeddingEvaluator()
    
    # Load fine-tuned model if available
    if os.path.exists('models/best_sales_embedding_model.pt'):
        evaluator.load_fine_tuned_model('models/best_sales_embedding_model.pt')
    
    # Run evaluation
    results = evaluator.evaluate_embeddings(train_df, test_df)
    
    # Generate report
    report = evaluator.generate_report()
    print(report)
    
    # Plot comparison
    evaluator.plot_comparison('results/embedding_comparison.png')
    
    # Save results
    evaluator.save_results('results/evaluation_results.pkl')

if __name__ == "__main__":
    main() 