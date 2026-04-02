import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import json

LABEL2ID = {
    'BACKGROUND': 0,
    'OBJECTIVE': 1,
    'METHODS': 2,
    'RESULTS': 3,
    'CONCLUSIONS': 4
}

MAX_LENGTH = 256
BATCH_SIZE = 32


class PubMedRCTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
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
    """Parse PubMed RCT dataset file."""
    data = []
    current_abstract = None
    current_label = None
    current_sentences = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('###'):
                if current_abstract is not None and current_sentences:
                    for sentence, label in current_sentences:
                        data.append({
                            'abstract_id': current_abstract,
                            'sentence': sentence,
                            'label': label
                        })
                current_abstract = line.replace('###', '').strip()
                current_sentences = []
            elif line.startswith('BACKGROUND:') or line.startswith('OBJECTIVE:') or \
                 line.startswith('METHODS:') or line.startswith('RESULTS:') or \
                 line.startswith('CONCLUSIONS:'):
                parts = line.split(maxsplit=1)
                current_label = parts[0]
                if len(parts) > 1:
                    current_sentences.append((parts[1].strip(), current_label))
            elif current_label and line:
                current_sentences.append((line, current_label))
        
        if current_abstract is not None and current_sentences:
            for sentence, label in current_sentences:
                data.append({
                    'abstract_id': current_abstract,
                    'sentence': sentence,
                    'label': label
                })
    
    return pd.DataFrame(data)


def load_from_csv(filepath):
    """Load data from CSV file."""
    df = pd.read_csv(filepath)
    df = df.rename(columns={'target': 'label'})
    df['label'] = df['label'].map(LABEL2ID)
    df = df.rename(columns={'abstract_text': 'sentence'})
    return df


def preprocess_data(input_path, output_dir, model_name='allenai/scibert_scivocab_uncased', max_samples=200000):
    """Main preprocessing function."""
    print("Loading and preprocessing data...")
    
    if input_path.endswith('.csv'):
        df = load_from_csv(input_path)
    else:
        df = parse_pubmed_file(input_path)
    
    print(f"Total samples: {len(df)}")
    
    if max_samples is not None:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Using subset: {max_samples} samples")
    
    unique_abstracts = df['abstract_id'].unique()
    print(f"Total unique abstracts: {len(unique_abstracts)}")
    
    train_abs, temp_abs = train_test_split(unique_abstracts, test_size=0.2, random_state=42)
    val_abs, test_abs = train_test_split(temp_abs, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_abs)} abstracts, Val: {len(val_abs)}, Test: {len(test_abs)}")
    
    df_train = df[df['abstract_id'].isin(train_abs)].reset_index(drop=True)
    df_val = df[df['abstract_id'].isin(val_abs)].reset_index(drop=True)
    df_test = df[df['abstract_id'].isin(test_abs)].reset_index(drop=True)
    
    print(f"Train samples: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    print("\nLabel distribution:")
    for label in LABEL2ID.keys():
        count = (df_train['label'] == LABEL2ID[label]).sum()
        print(f"  {label}: {count} ({100*count/len(df_train):.2f}%)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    df_train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    label_dist = df_train['label'].value_counts().sort_index().to_dict()
    with open(os.path.join(output_dir, 'label_distribution.json'), 'w') as f:
        json.dump(label_dist, f)
    
    print(f"\nData saved to {output_dir}")
    print("Preprocessing complete!")
    
    return df_train, df_val, df_test


def create_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    """Create dataloaders."""
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


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input data file', default='data/PubMed_200k_RCT/train.csv')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--model', type=str, default='allenai/scibert_scivocab_uncased', help='Model name for tokenizer')
    parser.add_argument('--max-samples', type=int, default=50000, help='Max samples to use')
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output, args.model, args.max_samples)