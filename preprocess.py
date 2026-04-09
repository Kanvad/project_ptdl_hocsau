# PubMed RCT Dataset Preprocessing Script
# Parses raw PubMed RCT data and creates train/val/test splits for model training

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import json

# Label mapping for 5-class classification
LABEL2ID = {
    'BACKGROUND': 0,
    'OBJECTIVE': 1,
    'METHODS': 2,
    'RESULTS': 3,
    'CONCLUSIONS': 4
}

# Default hyperparameters
MAX_LENGTH = 256
BATCH_SIZE = 32


class PubMedRCTDataset(Dataset):
    """PyTorch Dataset for PubMed RCT classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        """Initialize dataset with texts, labels, tokenizer, and max sequence length."""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def parse_pubmed_file(filepath):
    """Parse PubMed RCT dataset file in standard format.
    
    Expected format:
    ###12345
    BACKGROUND: sentence1
    sentence2
    METHODS: sentence3
    ...
    
    Args:
        filepath: Path to the input file
        
    Returns:
        DataFrame with columns: abstract_id, sentence, label
    """
    data = []
    current_abstract = None
    current_label = None
    current_sentences = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # New abstract starts with ###
            if line.startswith('###'):
                # Save previous abstract's sentences
                if current_abstract is not None and current_sentences:
                    for sentence, label in current_sentences:
                        data.append({
                            'abstract_id': current_abstract,
                            'sentence': sentence,
                            'label': label
                        })
                # Start new abstract
                current_abstract = line.replace('###', '').strip()
                current_sentences = []
            # Label header lines (e.g., "BACKGROUND:", "METHODS:")
            elif line.startswith('BACKGROUND:') or line.startswith('OBJECTIVE:') or \
                 line.startswith('METHODS:') or line.startswith('RESULTS:') or \
                 line.startswith('CONCLUSIONS:'):
                parts = line.split(maxsplit=1)
                current_label = parts[0]
                if len(parts) > 1:
                    current_sentences.append((parts[1].strip(), current_label))
            # Continuation of previous label
            elif current_label and line:
                current_sentences.append((line, current_label))
    
    # Handle last abstract
    if current_abstract is not None and current_sentences:
        for sentence, label in current_sentences:
            data.append({
                'abstract_id': current_abstract,
                'sentence': sentence,
                'label': label
            })
    
    return pd.DataFrame(data)


def load_from_csv(filepath):
    """Load data from CSV file with specific column names.
    
    Expected columns: abstract_text, target
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with columns: sentence, label
    """
    df = pd.read_csv(filepath)
    # Map 'target' column to 'label' and apply label encoding
    df = df.rename(columns={'target': 'label'})
    df['label'] = df['label'].map(LABEL2ID)
    # Rename abstract_text to sentence
    df = df.rename(columns={'abstract_text': 'sentence'})
    return df


def preprocess_data(input_path, output_dir, model_name='allenai/scibert_scivocab_uncased', max_samples=200000):
    """Main preprocessing function.
    
    Loads data, splits into train/val/test (80/10/10) by abstract,
    and saves to CSV files.
    
    Args:
        input_path: Path to input data (CSV or text file)
        output_dir: Directory to save processed data
        model_name: Model name for tokenizer (used for reference)
        max_samples: Maximum number of samples to use (None for all)
        
    Returns:
        Tuple of (df_train, df_val, df_test)
    """
    print("Loading and preprocessing data...")
    
    # Load data from CSV or parse from text file
    if input_path.endswith('.csv'):
        df = load_from_csv(input_path)
    else:
        df = parse_pubmed_file(input_path)
    
    print(f"Total samples: {len(df)}")
    
    # Optionally limit sample count for faster processing
    if max_samples is not None:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Using subset: {max_samples} samples")
    
    # Get unique abstracts for proper splitting (avoid data leakage)
    unique_abstracts = df['abstract_id'].unique()
    print(f"Total unique abstracts: {len(unique_abstracts)}")
    
    # Split abstracts: 80% train, 10% val, 10% test
    train_abs, temp_abs = train_test_split(unique_abstracts, test_size=0.2, random_state=42)
    val_abs, test_abs = train_test_split(temp_abs, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_abs)} abstracts, Val: {len(val_abs)}, Test: {len(test_abs)}")
    
    # Create dataframes by filtering on abstract IDs
    df_train = df[df['abstract_id'].isin(train_abs)].reset_index(drop=True)
    df_val = df[df['abstract_id'].isin(val_abs)].reset_index(drop=True)
    df_test = df[df['abstract_id'].isin(test_abs)].reset_index(drop=True)
    
    print(f"Train samples: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Print label distribution
    print("\nLabel distribution:")
    for label in LABEL2ID.keys():
        count = (df_train['label'] == LABEL2ID[label]).sum()
        print(f"  {label}: {count} ({100*count/len(df_train):.2f}%)")
    
    # Save processed data to output directory
    os.makedirs(output_dir, exist_ok=True)
    
    df_train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Save label distribution for reference
    label_dist = df_train['label'].value_counts().sort_index().to_dict()
    with open(os.path.join(output_dir, 'label_distribution.json'), 'w') as f:
        json.dump(label_dist, f)
    
    print(f"\nData saved to {output_dir}")
    print("Preprocessing complete!")
    
    return df_train, df_val, df_test


def create_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    """Create PyTorch DataLoaders for train/val/test sets.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        tokenizer: Tokenizer for encoding text
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = PubMedRCTDataset(
        train_df['sentence'].values,
        train_df['label'].values,
        tokenizer,
        max_length
    )
    
    val_dataset = PubMedRCTDataset(
        val_df['sentence'].values,
        val_df['label'].values,
        tokenizer,
        max_length
    )
    
    test_dataset = PubMedRCTDataset(
        test_df['sentence'].values,
        test_df['label'].values,
        tokenizer,
        max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# =============================================================================
# Main execution: Run preprocessing from command line
# =============================================================================
import argparse
    
parser = argparse.ArgumentParser(description='Preprocess PubMed RCT dataset')
parser.add_argument('--input', type=str, 
                    default='data/PubMed_200k_RCT/train.csv',
                    help='Path to input data file')
parser.add_argument('--output', type=str, default='data',
                    help='Output directory for processed data')
parser.add_argument('--model', type=str, 
                    default='allenai/scibert_scivocab_uncased',
                    help='Model name for tokenizer')
parser.add_argument('--max-samples', type=int, default=50000,
                    help='Maximum number of samples to use')
args = parser.parse_args()

preprocess_data(args.input, args.output, args.model, args.max_samples)