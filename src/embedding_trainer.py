import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle

class SalesConversationDataset(Dataset):
    def __init__(self, conversations: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.conversations = conversations
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = str(self.conversations[idx])
        label = self.labels[idx]
        
        # Tokenize conversation
        encoding = self.tokenizer(
            conversation,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ContrastiveEmbeddingModel(nn.Module):
    def __init__(self, base_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 embedding_dim: int = 384, num_classes: int = 2):
        super().__init__()
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, input_ids, attention_mask):
        # Get embeddings from base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        # Get projected embeddings for contrastive learning
        projected_embeddings = self.projection_head(mean_embeddings)
        
        # Get classification logits
        classification_logits = self.classifier(mean_embeddings)
        
        return mean_embeddings, projected_embeddings, classification_logits

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels):
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(mask.size(0), device=mask.device)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = torch.sum(exp_sim * (1 - torch.eye(exp_sim.size(0), device=exp_sim.device)), dim=1)
        
        positive_pairs = torch.sum(exp_sim * mask, dim=1)
        loss = -torch.log(positive_pairs / (sum_exp_sim + 1e-8))
        
        return torch.mean(loss)

class SalesEmbeddingTrainer:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = ContrastiveEmbeddingModel(model_name)
        self.model.to(self.device)
        
        # Loss functions
        self.contrastive_loss = ContrastiveLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        
    def prepare_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, batch_size: int = 16):
        """Prepare data loaders for training"""
        
        # Create datasets
        train_dataset = SalesConversationDataset(
            train_df['conversation'].tolist(),
            train_df['conversion_label'].tolist(),
            self.model.tokenizer
        )
        
        test_dataset = SalesConversationDataset(
            test_df['conversation'].tolist(),
            test_df['conversion_label'].tolist(),
            self.model.tokenizer
        )
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Test batches: {len(self.test_loader)}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_contrastive_loss = 0
        total_classification_loss = 0
        
        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            embeddings, projected_embeddings, classification_logits = self.model(input_ids, attention_mask)
            
            # Compute losses
            contrastive_loss = self.contrastive_loss(projected_embeddings, labels)
            classification_loss = self.classification_loss(classification_logits, labels)
            
            # Combined loss
            total_batch_loss = contrastive_loss + classification_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_classification_loss += classification_loss.item()
        
        return {
            'total_loss': total_loss / len(self.train_loader),
            'contrastive_loss': total_contrastive_loss / len(self.train_loader),
            'classification_loss': total_classification_loss / len(self.train_loader)
        }
    
    def evaluate(self):
        """Evaluate model on test set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                embeddings, projected_embeddings, classification_logits = self.model(input_ids, attention_mask)
                
                # Compute loss
                contrastive_loss = self.contrastive_loss(projected_embeddings, labels)
                classification_loss = self.classification_loss(classification_logits, labels)
                total_loss += (contrastive_loss + classification_loss).item()
                
                # Get predictions
                predictions = torch.argmax(classification_logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': total_loss / len(self.test_loader)
        }
    
    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
              epochs: int = 5, batch_size: int = 16):
        """Full training pipeline"""
        
        print("Preparing data...")
        self.prepare_data(train_df, test_df, batch_size)
        
        print("Starting training...")
        best_f1 = 0
        training_history = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            print(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                  f"Contrastive: {train_metrics['contrastive_loss']:.4f}, "
                  f"Classification: {train_metrics['classification_loss']:.4f}")
            
            # Evaluate
            eval_metrics = self.evaluate()
            print(f"Eval - Accuracy: {eval_metrics['accuracy']:.4f}, "
                  f"F1: {eval_metrics['f1']:.4f}, "
                  f"Loss: {eval_metrics['loss']:.4f}")
            
            # Save best model
            if eval_metrics['f1'] > best_f1:
                best_f1 = eval_metrics['f1']
                self.save_model('models/best_sales_embedding_model.pt')
                print(f"New best model saved with F1: {best_f1:.4f}")
            
            # Record metrics
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['total_loss'],
                'eval_loss': eval_metrics['loss'],
                'eval_accuracy': eval_metrics['accuracy'],
                'eval_f1': eval_metrics['f1']
            })
        
        return training_history
    
    def save_model(self, path: str):
        """Save the trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'embedding_dim': self.model.embedding_dim
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {path}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts"""
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.model.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Get embeddings
                embeddings, _, _ = self.model(input_ids, attention_mask)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def predict_conversion(self, texts: List[str]) -> List[Dict]:
        """Predict conversion probability for texts"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.model.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Get prediction
                _, _, classification_logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(classification_logits, dim=1)
                
                predictions.append({
                    'conversion_probability': probabilities[0][1].item(),
                    'predicted_label': torch.argmax(classification_logits, dim=1).item()
                })
        
        return predictions

def main():
    """Example usage"""
    # Load data
    train_df = pd.read_csv('data/train_conversations.csv')
    test_df = pd.read_csv('data/test_conversations.csv')
    
    # Initialize trainer
    trainer = SalesEmbeddingTrainer()
    
    # Train model
    history = trainer.train(train_df, test_df, epochs=3, batch_size=8)
    
    # Test predictions
    sample_texts = [
        "I'm very interested in this solution and would like to move forward with implementation",
        "I need to think about it and discuss with my team before making any decisions"
    ]
    
    predictions = trainer.predict_conversion(sample_texts)
    for text, pred in zip(sample_texts, predictions):
        print(f"Text: {text}")
        print(f"Conversion Probability: {pred['conversion_probability']:.3f}")
        print(f"Predicted Label: {pred['predicted_label']}")
        print("-" * 50)

if __name__ == "__main__":
    main() 