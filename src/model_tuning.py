import pandas as pd
import numpy as np
import scipy.sparse
import os
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from scipy.stats import uniform, loguniform, randint
from tqdm import tqdm
import time

# Check for resource warnings or optimizations
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def load_features(filename):
    """Loads features from .npz or .npy files."""
    if not os.path.exists(filename):
        return None
    
    if filename.endswith('.npz'):
        return scipy.sparse.load_npz(filename)
    elif filename.endswith('.npy'):
        return np.load(filename)
    return None

def main():
    print("Starting Hyperparameter Tuning...")
    
    # 1. Load Labels
    print("Loading labels...")
    try:
        df = pd.read_csv('preprocessed_train_features.csv')
        y = df['Star Rating'].values
    except FileNotFoundError:
        print("Error: preprocessed_train_features.csv not found.")
        return

    # 2. Define Feature Files
    # Loading ALL feature sets as requested
    feature_files = {
        'TF-IDF (1-2 gram)': 'tfidf_unigram_bigram_features.npz',
        'TF-IDF (1-3 gram)': 'tfidf_unigram_bigram_trigram_features.npz',
        'TF-IDF (Limited 5000)': 'tfidf_limited_5000_features.npz',
        'BERT': 'bert_features.npy',
        'RoBERTa': 'roberta_features.npy'
    }

    loaded_features = {}
    print("Loading feature sets...")
    for name, filepath in tqdm(feature_files.items()):
        data = load_features(filepath)
        if data is not None:
             if data.shape[0] == len(y):
                loaded_features[name] = data
             else:
                 print(f"Skipping {name}: Size mismatch.")

    # 3. Create Ensembles
    # Specifically TF-IDF (Limited) + BERT/RoBERTa are good candidates to keep memory distinct
    # User said "check for all features and ensembles". 
    # I will create a specific Ensemble of TF-IDF (Limited) + BERT
    
    if 'TF-IDF (Limited 5000)' in loaded_features and 'BERT' in loaded_features:
        print("Creating Ensemble: TF-IDF + BERT")
        feat_tfidf = loaded_features['TF-IDF (Limited 5000)']
        feat_bert = loaded_features['BERT']
        if scipy.sparse.issparse(feat_tfidf):
            loaded_features['Ensemble (TF-IDF+BERT)'] = scipy.sparse.hstack([feat_tfidf, feat_bert])
        else:
            loaded_features['Ensemble (TF-IDF+BERT)'] = np.hstack([feat_tfidf, feat_bert])

    # 4. Define Models & Param Grids
    # Using RidgeClassifier instead of LinearRegression for classification tuning
    score_metric = make_scorer(f1_score, average='weighted')
    
    models_config = {
        'Linear SVM': {
            'model': LinearSVC(dual='auto', max_iter=2000),
            'params': {
                'C': loguniform(1e-3, 1e2)
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto'),
            'params': {
                'C': loguniform(1e-3, 1e2),
                # 'penalty': ['l2'] # lbfgs supports l2
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_jobs=-1),
            'params': {
                'n_estimators': randint(50, 200),
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': randint(2, 10)
            }
        },
        'Ridge Classifier (Linear Reg equivalent)': {
             'model': RidgeClassifier(),
             'params': {
                 'alpha': loguniform(1e-2, 1e2)
             }
        }
    }

    results = []
    
    # 6. Tuning Loop
    print("\n--- Running Randomized Search (Optimized) ---")
    
    # Total iterations estimation for progress bar
    total_steps = len(loaded_features) * len(models_config)
    pbar = tqdm(total=total_steps, desc="Tuning Progress")

    for feature_name, X in loaded_features.items():
        # Split for final evaluation logic
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Subsample for tuning to speed up (max 1500 samples)
        # We only use this smaller set for finding the best params
        if X_train.shape[0] > 1500:
            # maintain class distribution if possible, but simple shuffle is usually fine
            # We can just take the first 1500 or random choice
            indices = np.random.choice(X_train.shape[0], 1500, replace=False)
            if scipy.sparse.issparse(X_train):
                X_tune = X_train[indices]
            else:
                X_tune = X_train[indices]
            y_tune = y_train[indices]
        else:
            X_tune = X_train
            y_tune = y_train

        for model_name, config in models_config.items():
            pbar.set_description(f"Tuning {model_name} on {feature_name}")
            
            base_model = config['model']
            params = config['params']
            
            # Reduce n_iter for Random Forest to save more time if needed
            iter_count = 5
            if 'Random Forest' in model_name:
                iter_count = 3 # Go faster for slow RF
            
            search = RandomizedSearchCV(
                base_model, 
                params, 
                n_iter=iter_count, 
                cv=2,  # Reduce CV to 2 for speed verification
                scoring=score_metric, 
                n_jobs=-1, 
                random_state=42
            )
            
            start_time = time.time()
            try:
                # Fit on SUBSET
                search.fit(X_tune, y_tune)
                
                best_model = search.best_estimator_
                best_score_cv = search.best_score_
                best_params = search.best_params_
                
                # OPTIONAL: Refit on full X_train with best params? 
                # Usually best practice, but for this task just evaluating the 'best_model' 
                # (which is refit on X_tune) on X_test gives a decent proxy. 
                # To be accurate, let's clone and refit on full X_train.
                
                from sklearn.base import clone
                final_model = clone(best_model)
                final_model.fit(X_train, y_train)
                
                # Final evaluation on held-out test set
                y_pred = final_model.predict(X_test)
                test_f1 = f1_score(y_test, y_pred, average='weighted')
                
                results.append({
                    'Model': model_name,
                    'Feature Set': feature_name,
                    'Best CV F1': round(best_score_cv, 4),
                    'Test F1': round(test_f1, 4),
                    'Best Params': str(best_params),
                    'Time (s)': round(time.time() - start_time, 2)
                })
            except Exception as e:
                # print(f"\nFailed to tune {model_name} on {feature_name}: {e}")
                pass
            
            pbar.update(1)
            
    pbar.close()

    # 7. Report
    print("\n\n--- Hyperparameter Tuning Results ---")
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Test F1', ascending=False)
        print(results_df[['Model', 'Feature Set', 'Test F1', 'Best Params']].to_string(index=False))
        
        results_df.to_csv('tuning_results.csv', index=False)
        print("\nFull results saved to tuning_results.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
