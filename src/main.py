import preprocessing_dataset
import feature_extraction
import model_training
import model_tuning
import predict_test
import time

def main():
    print("==================================================")
    print("      AUTOMATED ML PIPELINE: RATING PREDICTION    ")
    print("==================================================")
    
    total_start = time.time()

    # 1. Preprocessing
    print("\n[STEP 1/5] PREPROCESSING DATASET")
    print("-" * 30)
    preprocessing_dataset.main()
    
    # 2. Feature Extraction
    print("\n[STEP 2/5] FEATURE EXTRACTION")
    print("-" * 30)
    feature_extraction.main()
    
    # 3. Model Training (Baselines)
    print("\n[STEP 3/5] BASELINE MODEL TRAINING & VALIDATION")
    print("-" * 30)
    model_training.main()
    
    # 4. Hyperparameter Tuning
    print("\n[STEP 4/5] HYPERPARAMETER TUNING")
    print("-" * 30)
    # This can be time consuming.
    print("Running optimization to find best model...")
    model_tuning.main()
    
    # 5. Final Prediction
    print("\n[STEP 5/5] FINAL PREDICTION")
    print("-" * 30)
    predict_test.main()
    
    total_end = time.time()
    print("\n==================================================")
    print(f"PIPELINE COMPLETED in {(total_end - total_start)/60:.2f} minutes.")
    print("Results stored in: submission.csv")
    print("==================================================")

if __name__ == "__main__":
    main()
