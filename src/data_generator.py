import random
import pandas as pd
from typing import List, Dict, Tuple
import numpy as np

class SalesDataGenerator:
    def __init__(self):
        # High conversion conversation patterns
        self.high_conversion_patterns = [
            "I'm very interested in this solution for our company",
            "This looks like exactly what we need",
            "When can we get started with implementation?",
            "What's the next step in the process?",
            "I'd like to schedule a follow-up meeting",
            "Can you send me a proposal?",
            "This fits perfectly with our budget",
            "I need to discuss this with my team, but I'm excited",
            "How soon can we see results?",
            "I'm ready to move forward",
            "This addresses all our pain points",
            "I can see the ROI clearly",
            "Let's schedule a demo for the team",
            "I'm convinced this is the right choice",
            "What are the implementation timelines?"
        ]
        
        # Low conversion conversation patterns
        self.low_conversion_patterns = [
            "I need to think about it",
            "This is too expensive for us right now",
            "We're not ready to make any decisions",
            "I'm just gathering information",
            "We have other priorities at the moment",
            "I need to compare with other vendors",
            "The timing isn't right",
            "I'm not sure this fits our needs",
            "We don't have budget allocated for this",
            "I'll get back to you later",
            "This seems too complex for our team",
            "We're happy with our current solution",
            "I need approval from upper management",
            "Let me discuss with my colleagues first",
            "I'm not the decision maker"
        ]
        
        # Common conversation starters and fillers
        self.conversation_starters = [
            "Thank you for taking the time to speak with me today",
            "I understand you're looking for a solution to help with",
            "Let me tell you about our product and how it can help",
            "I'd love to learn more about your current challenges",
            "Based on our previous conversation"
        ]
        
        self.objection_responses = [
            "I understand your concern about",
            "That's a great question, let me address that",
            "Many of our clients had similar concerns initially",
            "I can see why that would be important to you",
            "Let me explain how we handle that"
        ]
        
        # Sales-specific vocabulary
        self.sales_terms = [
            "ROI", "implementation", "solution", "investment", "value proposition",
            "pain points", "decision maker", "budget", "timeline", "proposal",
            "demo", "trial", "contract", "pricing", "features", "benefits",
            "competitive advantage", "cost savings", "efficiency", "productivity"
        ]
    
    def generate_conversation(self, conversion_label: int) -> str:
        """Generate a realistic sales conversation based on conversion label"""
        conversation_parts = []
        
        # Start with opener
        conversation_parts.append(random.choice(self.conversation_starters))
        
        # Add 3-5 conversation exchanges
        num_exchanges = random.randint(3, 5)
        
        for i in range(num_exchanges):
            if conversion_label == 1:  # High conversion
                # Mix of positive responses and some objections that get resolved
                if random.random() < 0.7:  # 70% positive
                    conversation_parts.append(random.choice(self.high_conversion_patterns))
                else:  # 30% objections that get addressed
                    conversation_parts.append(random.choice(self.objection_responses))
                    conversation_parts.append(random.choice(self.high_conversion_patterns))
            else:  # Low conversion
                # Mix of neutral and negative responses
                if random.random() < 0.8:  # 80% negative/neutral
                    conversation_parts.append(random.choice(self.low_conversion_patterns))
                else:  # 20% slightly positive but still ends negative
                    conversation_parts.append(random.choice(self.high_conversion_patterns))
                    conversation_parts.append(random.choice(self.low_conversion_patterns))
        
        # Add some sales-specific terms
        if random.random() < 0.6:
            term = random.choice(self.sales_terms)
            conversation_parts.append(f"Regarding the {term}, I think it's important to consider")
        
        return " ".join(conversation_parts)
    
    def generate_dataset(self, num_samples: int = 1000) -> pd.DataFrame:
        """Generate a balanced dataset of sales conversations"""
        conversations = []
        labels = []
        
        # Generate balanced dataset
        for i in range(num_samples):
            # 50/50 split between high and low conversion
            label = 1 if i < num_samples // 2 else 0
            conversation = self.generate_conversation(label)
            
            conversations.append(conversation)
            labels.append(label)
        
        # Shuffle the dataset
        combined = list(zip(conversations, labels))
        random.shuffle(combined)
        conversations, labels = zip(*combined)
        
        # Create DataFrame
        df = pd.DataFrame({
            'conversation': conversations,
            'conversion_label': labels,
            'conversation_id': range(len(conversations))
        })
        
        return df
    
    def add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic metadata to conversations"""
        industries = ['Technology', 'Healthcare', 'Finance', 'Retail', 'Manufacturing']
        company_sizes = ['Small', 'Medium', 'Large', 'Enterprise']
        call_durations = np.random.normal(25, 8, len(df))  # Average 25 min calls
        
        df['industry'] = [random.choice(industries) for _ in range(len(df))]
        df['company_size'] = [random.choice(company_sizes) for _ in range(len(df))]
        df['call_duration_minutes'] = np.clip(call_durations, 5, 60).astype(int)
        df['sales_rep'] = [f"Rep_{random.randint(1, 10)}" for _ in range(len(df))]
        
        return df

def main():
    """Generate and save sample dataset"""
    generator = SalesDataGenerator()
    
    # Generate training dataset
    print("Generating training dataset...")
    train_df = generator.generate_dataset(num_samples=800)
    train_df = generator.add_metadata(train_df)
    
    # Generate test dataset
    print("Generating test dataset...")
    test_df = generator.generate_dataset(num_samples=200)
    test_df = generator.add_metadata(test_df)
    
    # Save datasets
    train_df.to_csv('data/train_conversations.csv', index=False)
    test_df.to_csv('data/test_conversations.csv', index=False)
    
    print(f"Generated {len(train_df)} training samples and {len(test_df)} test samples")
    print(f"Training set conversion rate: {train_df['conversion_label'].mean():.2%}")
    print(f"Test set conversion rate: {test_df['conversion_label'].mean():.2%}")
    
    # Show sample conversations
    print("\nSample High Conversion Conversation:")
    high_conv = train_df[train_df['conversion_label'] == 1].iloc[0]
    print(high_conv['conversation'])
    
    print("\nSample Low Conversion Conversation:")
    low_conv = train_df[train_df['conversion_label'] == 0].iloc[0]
    print(low_conv['conversation'])

if __name__ == "__main__":
    main() 