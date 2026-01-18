# App Review Rating Prediction

## 1. Problem Overview
The objective of this project is to build a robust machine learning pipeline to predict the **Star Rating (1-5)** of a mobile application based on user reviews. This is a **multi-class classification problem** where the input is unstructured text (the user review) and the output is an ordinal integer label representing user sentiment and satisfaction.

The dataset consists of user reviews with associated metadata. The core challenge lies in extracting meaningful sentiment and semantic signals from the raw text to accurately classify the reviews into one of the five rating categories.

## 2. Approach and Modeling Technique
We adopted a systematic approach involving rigorous preprocessing, diverse feature engineering, and robust model selection.

### Preprocessing Pipeline
Text data is noisy. We implemented a custom preprocessing pipeline (`preprocessing_dataset.py`) that includes:
-   **Cleaning**: Lowercasing, removing removing special characters/punctuation, and stripping whitespace.
-   **Tokenization**: Splitting text into individual words using NLTK.
-   **Smart Stopword Removal**: Standard stopword lists often remove words like "no" or "not", which flips sentiment (e.g., "not good" becomes "good"). We customized the removal process to **preserve negation words** to maintain sentiment integrity.
-   **Lemmatization**: Reducing words to their base form (e.g., "running" -> "run") to reduce feature sparsity.

### Modeling Techniques
We evaluated multiple algorithms ranging from simple linear baselines to ensemble methods:
-   **Linear Regression**: Used as a naive baseline (predictions rounded to nearest integer).
-   **Linear SVM (Support Vector Machine)**: Effective for high-dimensional sparse data (like TF-IDF). This proved to be the **best performing model**.
-   **Random Forest Classifier**: A non-linear tree-based ensemble to capture complex interactions.
-   **Logistic Regression**: A probabilistic linear baseline.

## 3. Feature Extraction Strategy
To interpret the text, we employed a hybrid feature extraction strategy combining statistical and deep learning methods:

### A. Statistical Features (TF-IDF)
We used **Term Frequency-Inverse Document Frequency (TF-IDF)** to capture the importance of words:
-   **N-grams**: We experimented with Unigrams (1-word), Bigrams (2-word sequences), and Trigrams (3-word sequences) to capture local context.
-   **Dimensionality Control**: We limited features (e.g., top 5000) to balance information gain vs. computational efficiency.

### B. Contextual Embeddings (Transformers)
We utilized state-of-the-art Transformer models to generate dense vector embeddings that capture deep semantic meaning:
-   **BERT (`bert-base-uncased`)**: Bidirectional Encoder Representations from Transformers.
-   **RoBERTa (`roberta-base`)**: A robustly optimized BERT variant.
We extracted the `[CLS]` token embedding from the user reviews to represent the entire sentence semantics.

### C. Feature Ensemble
Our best results came from **stacking** features: combining the sparse TF-IDF matrix with the dense BERT embeddings. This allowed the model to leverage both specific keyword signals (TF-IDF) and broad semantic understanding (BERT).

## 4. Validation Methodology
Given the potential for class imbalance in review ratings, we prioritized robust evaluation metrics and tuning strategies:

-   **Primary Metric**: **Weighted F1 Score**. This calculates the F1 score for each class and calculates a weighted average based on the number of samples per class. This is superior to simple accuracy for this domain.
-   **Cross-Validation**: We employed **K-Fold Cross-Validation** (via `RandomizedSearchCV`) during hyperparameter tuning to ensure the model generalizes well and is not overfitting to a specific data split.
-   **Hyperparameter Tuning**: We performed a randomized search over critical parameters:
    -   *SVM*: Regularization parameter `C`.
    -   *Random Forest*: `n_estimators`, `max_depth`, `min_samples_split`.
    -   *Logistic Regression*: Regularization strength `C`.

## 5. Instructions to Run the Code

The project is designed to be fully automated.

### Prerequisites
Ensure you have Python 3.x installed along with the required libraries:
```bash
pip install pandas numpy scipy scikit-learn torch transformers nltk tqdm
```

### Automatic Execution (Recommended)
To run the entire pipeline (Preprocessing -> Feature Extraction -> Training -> Tuning -> Prediction) sequentially:

```bash
python main.py
```

This master script will:
1.  Clean the dataset and generate `preprocessed_train_features.csv`.
2.  Extract TF-IDF and BERT features.
3.  Train baseline models and report initial metrics.
4.  Run hyperparameter tuning to find the optimal configuration.
5.  Use the best found model to generate predictions for `test.csv`.
6.  Save the final predictions to **`predictions.csv`**.

### Manual Step-by-Step Execution
If you wish to run specific stages individually:

1.  **Preprocessing**: `python preprocessing_dataset.py`
2.  **Feature Extraction**: `python feature_extraction.py`
3.  **Model Training**: `python model_training.py`
4.  **Hyperparameter Tuning**: `python model_tuning.py`
5.  **Prediction**: `python predict_test.py`

## 6. Results & Performance

We successfully ran the automated pipeline on the provided dataset.

-   **Total Pipeline Execution Time**: ~27.6 minutes.
-   **Best Model Identified**: Linear SVM on Ensemble Features (TF-IDF + BERT).
-   **Best Hyperparameters**: `{'C': 0.075}`
-   **Performance Metric (Weighted F1)**: **0.7106**

### Top 5 Model Configurations

The following table summarizes the performance of the top tuning results:

| Rank | Model | Feature Set | Weighted F1 Score |
|------|-------|-------------|-------------------|
| **1** | **Linear SVM** | **Ensemble (TF-IDF + BERT)** | **0.7106** |
| 2 | Logistic Regression | TF-IDF (Limited 5000) | 0.7105 |
| 3 | Logistic Regression | RoBERTa (Transformer) | 0.7100 |
| 4 | Ridge Classifier | Ensemble (TF-IDF + BERT) | 0.7098 |
| 5 | Linear SVM | TF-IDF (Limited 5000) | 0.7092 |

The tuning process highlighted that combining traditional statistical features (TF-IDF) with modern deep learning embeddings (BERT) yields the highest performance, outperforming either method individually. The Linear SVM efficiently handled this high-dimensional feature space.
