import pandas as pd
import numpy as np
import scipy.sparse
import torch
import re
import string
import ast
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from tqdm import tqdm

# Ensure NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def preprocess_text(text):
    """
    Same preprocessing as in preprocessing_dataset.py
    """
    if not isinstance(text, str):
        return ""

    # 1. Lower case
    text = text.lower()
    
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Remove whitespaces
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    # 4. Tokenization
    tokens = word_tokenize(text)
    
    # 5. Remove stopwords but keep negations
    stop_words = stopwords.words('english')
    negations = ['no', 'not', 'nor', 'neither', 'never', 'none', "don't", "aren't", "couldn't", "didn't", 
                 "doesn't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't", 
                 "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't"]
    
    final_stop_words = [sw for sw in stop_words if sw not in negations]
            
    new_tokens = [word for word in tokens if word not in final_stop_words]
    
    # 6. Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemma_tokens = [lemmatizer.lemmatize(word) for word in new_tokens]
    
    return " ".join(lemma_tokens)

def extract_transformer_features(texts, model_name='bert-base-uncased', batch_size=32):
    """
    Extracts features using BERT or RoBERTa.
    """
    print(f"Loading {model_name} model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        if 'roberta' in model_name:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaModel.from_pretrained(model_name)
        else:
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            
        model.to(device)
        model.eval()
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"Extracting features for {len(texts)} samples...")
        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc=f"{model_name} Extraction"):
            batch_texts = texts[i:i+batch_size]
            
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                outputs = model(**encoded_input)
                # For BERT/RoBERTa, use CLS token (index 0)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(cls_embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def get_best_model_config():
    """
    Reads tuning_results.csv and returns the best configuration.
    """
    try:
        df = pd.read_csv('tuning_results.csv')
        if df.empty:
            return None
        
        # Sort by Test F1 descending
        best_row = df.sort_values(by='Test F1', ascending=False).iloc[0]
        return best_row
    except FileNotFoundError:
        print("tuning_results.csv not found. Please run model_tuning.py first.")
        return None

def get_tfidf_config(feature_set_name):
    """
    Returns (ngram_range, max_features) based on string.
    """
    if "Limited 5000" in feature_set_name:
        return (1, 3), 5000 # Assuming limited uses 1-3
    elif "1-3 gram" in feature_set_name:
        return (1, 3), None
    elif "1-2 gram" in feature_set_name:
        return (1, 2), None
    else:
        # Default fallback
        return (1, 2), None

def main():
    # 1. Identify Best Model
    print("--- 1. Model Selection ---")
    best_config = get_best_model_config()
    if best_config is None:
        print("Could not identify best model. Exiting.")
        return

    model_name = best_config['Model']
    feature_set = best_config['Feature Set']
    best_params_str = best_config['Best Params']
    best_f1 = best_config['Test F1']
    
    print(f"Best Configuration Found:")
    print(f"  Model: {model_name}")
    print(f"  Feature Set: {feature_set}")
    print(f"  Test F1: {best_f1}")
    print(f"  Params: {best_params_str}")
    
    # Parse Params
    try:
        params = ast.literal_eval(best_params_str)
    except:
        print("Error parsing parameters. Using defaults.")
        params = {}

    # 2. Load Data
    print("\n--- 2. Loading Data ---")
    try:
        df_train = pd.read_csv('preprocessed_train_features.csv')
        df_test = pd.read_csv('test.csv') # Raw test data
        
        # Ensure text columns
        df_train['Review Text'] = df_train['Review Text'].fillna("")
        y_train = df_train['Star Rating'].values
        
        df_test['Review Text'] = df_test['Review Text'].fillna("")
        df_test['Review Title'] = df_test['Review Title'].fillna("")
    except FileNotFoundError:
        print("Data files not found.")
        return

    # 3. Feature Engineering (Dynamic)
    print("\n--- 3. Feature Extraction ---")
    
    # Preprocess Test Text
    print("Preprocessing Test Corpus...")
    test_texts_raw = df_test['Review Text'].apply(preprocess_text).tolist()
    train_texts = df_train['Review Text'].tolist()
    
    X_train_final = None
    X_test_final = None
    
    # Logic for Feature Sets
    use_tfidf = "TF-IDF" in feature_set
    use_bert = "BERT" in feature_set and "RoBERTa" not in feature_set
    use_roberta = "RoBERTa" in feature_set
    
    # A. TF-IDF
    if use_tfidf:
        ngram, max_feat = get_tfidf_config(feature_set)
        print(f"Generating TF-IDF (ngram={ngram}, max_features={max_feat})...")
        vectorizer = TfidfVectorizer(ngram_range=ngram, max_features=max_feat)
        
        X_train_tfidf = vectorizer.fit_transform(train_texts)
        X_test_tfidf = vectorizer.transform(test_texts_raw)
        
        X_train_final = X_train_tfidf
        X_test_final = X_test_tfidf
        
    # B. Transformer
    transformer_model = None
    if use_bert:
        transformer_model = 'bert-base-uncased'
    elif use_roberta:
        transformer_model = 'roberta-base'
        
    if transformer_model:
        print(f"Generating Transformer Features ({transformer_model})...")
        # Check if we can load pre-computed TRAIN features
        train_feat_file = 'bert_features.npy' if 'bert' in transformer_model else 'roberta_features.npy'
        
        try:
             X_train_trans = np.load(train_feat_file)
             print(f"Loaded existing {train_feat_file}.")
        except FileNotFoundError:
             print(f"Could not find {train_feat_file}, extracting from scratch (this may take time)...")
             X_train_trans = extract_transformer_features(train_texts, model_name=transformer_model)

        # Extract TEST features (Always fresh)
        X_test_trans = extract_transformer_features(test_texts_raw, model_name=transformer_model)
        
        if X_train_final is not None:
            # Stack with TF-IDF
            print("Stacking TF-IDF and Transformer features...")
            if scipy.sparse.issparse(X_train_final):
                X_train_final = scipy.sparse.hstack([X_train_final, X_train_trans])
                X_test_final = scipy.sparse.hstack([X_test_final, X_test_trans])
            else:
                 X_train_final = np.hstack([X_train_final, X_train_trans])
                 X_test_final = np.hstack([X_test_final, X_test_trans])
        else:
            X_train_final = X_train_trans
            X_test_final = X_test_trans

    print(f"Final Train Shape: {X_train_final.shape}")
    print(f"Final Test Shape: {X_test_final.shape}")

    # 4. Model Training
    print("\n--- 4. Training Best Model ---")
    clf = None
    
    if "Linear SVM" in model_name:
        clf = LinearSVC(**params, dual='auto', max_iter=2000)
    elif "Logistic Regression" in model_name:
        clf = LogisticRegression(**params, max_iter=1000, solver='lbfgs', multi_class='auto')
    elif "Random Forest" in model_name:
        # Check if params has n_jobs, else add it
        if 'n_jobs' not in params:
            params['n_jobs'] = -1
        clf = RandomForestClassifier(**params)
    elif "Ridge Classifier" in model_name:
        clf = RidgeClassifier(**params)
        
    if clf is None:
        print(f"Unknown model type: {model_name}")
        return
        
    print(f"fitting {model_name}...")
    clf.fit(X_train_final, y_train)
    
    # 5. Prediction
    print("\n--- 5. Prediction ---")
    predictions = clf.predict(X_test_final)
    
    # 6. Save
    submission = pd.DataFrame({
        'id': df_test['id'],
        'Star Rating': predictions
    })
    
    submission.to_csv('predictions.csv', index=False)
    print("Success! predictions.csv created using the best tuned model.")
    print(submission.head())

if __name__ == "__main__":
    main()
