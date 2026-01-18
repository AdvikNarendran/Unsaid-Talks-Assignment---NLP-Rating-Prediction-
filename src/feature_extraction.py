import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel

import scipy.sparse

# check for gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def extract_tfidf_features(df, text_column, ngram_range=(1, 1), max_features=None, output_name="tfidf"):
    """
    Extracts TF-IDF features and saves them.
    """
    print(f"\n--- TF-IDF Extraction (N-gram: {ngram_range}, Max Features: {max_features}) ---")
    tfidf = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    # fill na just in case
    texts = df[text_column].fillna("").tolist()
    features = tfidf.fit_transform(texts)
    print(f"Shape: {features.shape}")
    
    # Save
    filename = f"{output_name}_features.npz"
    print(f"Saving to {filename}...")
    scipy.sparse.save_npz(filename, features)
    return features

def extract_transformer_features(df, text_column, model_name, batch_size=32, output_name="bert"):
    """
    Extracts features using a Transformer model (BERT/RoBERTa) and saves them.
    Uses the [CLS] token embedding.
    """
    print(f"\n--- Deep Learning Features ({model_name}) ---")
    
    try:
        if 'roberta' in model_name:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaModel.from_pretrained(model_name)
        else:
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            
        model.to(device)
        model.eval()
        
        texts = df[text_column].fillna("").tolist()
        all_embeddings = []
        
        # processing in batches
        total_batches = (len(texts) + batch_size - 1) // batch_size
        print(f"Processing {len(texts)} samples in {total_batches} batches...")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # tokenize
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                outputs = model(**encoded_input)
                # cls token is usually the first token (index 0)
                # last_hidden_state shape: (batch_size, seq_len, hidden_dim)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(cls_embeddings.cpu().numpy())
            
            if (i // batch_size) % 10 == 0:
                 print(f"Batch {i // batch_size + 1}/{total_batches} processed.")

        final_embeddings = np.vstack(all_embeddings)
        print(f"Feature extraction complete. Shape: {final_embeddings.shape}")
        
        # Save
        filename = f"{output_name}_features.npy"
        print(f"Saving to {filename}...")
        np.save(filename, final_embeddings)
        
        return final_embeddings

    except Exception as e:
        print(f"Error during transformer extraction: {e}")
        return None

def main():
    input_file = 'preprocessed_train_features.csv'
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # We will use 'Review Text' for feature extraction
    text_col = 'Review Text'
    if text_col not in df.columns:
        print(f"Column '{text_col}' not found in dataset.")
        return
    
    # 1. TF-IDF Unigram + Bigram
    extract_tfidf_features(df, text_col, ngram_range=(1, 2), output_name="tfidf_unigram_bigram")
    
    # 2. TF-IDF Unigram + Bigram + Trigram
    extract_tfidf_features(df, text_col, ngram_range=(1, 3), output_name="tfidf_unigram_bigram_trigram")
    
    # 3. TF-IDF + N-gram + Feature Limiting (Unigram+Bigram+Trigram, max 5000 features)
    extract_tfidf_features(df, text_col, ngram_range=(1, 3), max_features=5000, output_name="tfidf_limited_5000")
    
    # 4. BERT
    extract_transformer_features(df, text_col, model_name='bert-base-uncased', output_name="bert")
    
    # 5. RoBERTa
    extract_transformer_features(df, text_col, model_name='roberta-base', output_name="roberta")
    
    print("\nAll feature extractions completed and saved.")

if __name__ == "__main__":
    main()
