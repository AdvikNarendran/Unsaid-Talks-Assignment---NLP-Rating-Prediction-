import pandas as pd
import numpy as np
import scipy.sparse
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
from tqdm import tqdm
import time

def load_features(filename):
    """Loads features from .npz or .npy files."""
    if not os.path.exists(filename):
        return None
    
    print(f"Loading {filename}...")
    if filename.endswith('.npz'):
        return scipy.sparse.load_npz(filename)
    elif filename.endswith('.npy'):
        return np.load(filename)
    return None

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, feature_name):
    """
    Trains a model and evaluates it. 
    Returns a dictionary of metrics.
    """
    start_time = time.time()
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_raw = model.predict(X_test)
    
    # Handle Linear Regression (Classification by rounding)
    if isinstance(model, LinearRegression):
        y_pred = np.clip(np.round(y_pred_raw), 1, 5).astype(int)
    else:
        y_pred = y_pred_raw
        
    # Metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    
    end_time = time.time()
    
    return {
        'Model': model_name,
        'Feature Set': feature_name,
        'Weighted F1': f1,
        'Accuracy': acc,
        'Time (s)': round(end_time - start_time, 2)
    }

def main():
    # 1. Load Labels
    print("Loading labels from preprocessed_train_features.csv...")
    try:
        df = pd.read_csv('preprocessed_train_features.csv')
        y = df['Star Rating'].values
    except FileNotFoundError:
        print("Error: preprocessed_train_features.csv not found.")
        return

    # 2. Define Feature Files
    feature_files = {
        'TF-IDF (1-2 gram)': 'tfidf_unigram_bigram_features.npz',
        'TF-IDF (1-3 gram)': 'tfidf_unigram_bigram_trigram_features.npz',
        'TF-IDF (Limited 5000)': 'tfidf_limited_5000_features.npz',
        'BERT': 'bert_features.npy',
        'RoBERTa': 'roberta_features.npy'
    }

    # 3. Define Models
    # Note: Using LinearSVC for SVM as it's faster for text. 
    # Using LogisticRegression as a strong baseline alternative to RF/LinearReg.
    models = {
        'Linear Regression': LinearRegression(),
        'Linear SVM': LinearSVC(dual='auto', max_iter=1000), # dual='auto' handles n_samples > n_features better automatically
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1)
    }

    results = []

    # 4. Individual Feature Training
    print("\n--- Individual Feature Training ---")
    
    # Iterate over features with a progress bar
    for name, filepath in tqdm(feature_files.items(), desc="Feature Sets"):
        X = load_features(filepath)
        if X is None:
            print(f"Skipping {name} (file not found)")
            continue
            
        # Ensure X matches y length
        if X.shape[0] != len(y):
            print(f"Warning: Sample mismatch for {name}. X: {X.shape[0]}, y: {len(y)}. Skipping.")
            continue
            
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train all models on this feature set
        for model_name, model in tqdm(models.items(), desc=f"Models ({name})", leave=False):
            res = train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, name)
            results.append(res)

    # 5. Ensemble Feature Training (TF-IDF + Transformer)
    print("\n--- Ensemble Feature Training ---")
    # We will combine the best TF-IDF (usually Limited or 1-2 gram) with BERT if available.
    
    # Load specific features for ensemble
    tfidf_path = 'tfidf_limited_5000_features.npz' # choosing limit for memory safety in ensemble
    bert_path = 'bert_features.npy'
    
    X_tfidf = load_features(tfidf_path)
    X_bert = load_features(bert_path)
    
    if X_tfidf is not None and X_bert is not None:
        print("Creating Ensemble: TF-IDF (Limited) + BERT...")
        # Stack features
        if scipy.sparse.issparse(X_tfidf):
            # Convert bert to sparse if needed or stack sparse with dense (returns coo_matrix)
            # scipy.sparse.hstack handles sparse + dense
            X_ensemble = scipy.sparse.hstack([X_tfidf, X_bert])
        else:
            X_ensemble = np.hstack([X_tfidf, X_bert])
            
        X_train, X_test, y_train, y_test = train_test_split(X_ensemble, y, test_size=0.2, random_state=42)
        
        for model_name, model in tqdm(models.items(), desc="Models (Ensemble: TF-IDF+BERT)"):
            res = train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, "Ensemble (TF-IDF+BERT)")
            results.append(res)
    else:
        print("Skipping Ensemble (missing files).")

    # 6. Report
    print("\n--- Final Results ---")
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Sort by F1 Score
        results_df = results_df.sort_values(by='Weighted F1', ascending=False)
        print(results_df.to_string(index=False))
        
        results_df.to_csv('model_training_results.csv', index=False)
        print("\nResults saved to model_training_results.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
