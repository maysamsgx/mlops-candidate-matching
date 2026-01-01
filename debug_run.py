import pandas as pd
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from src.features import SkillsEmbeddingTransformer
from src.modeling import CandidateMatcherEnsemble

def main():
    print("Debug: Starting...")
    data_path = "candidate_job_role_dataset.csv"
    if not os.path.exists(data_path):
        print("Dataset not found!")
        return

    print("Debug: Loading data...")
    df = pd.read_csv(data_path)
    
    # Filter rare classes for CV
    v = df['job_role'].value_counts()
    df = df[df['job_role'].isin(v[v >= 3].index)]
    
    print(f"Debug: Data shape {df.shape}")
    
    # 1. Validation (Skip for debug)
    
    # 2. Training Logic
    print("Debug: Preparing pipeline...")
    X = df[['candidate_id', 'skills', 'qualification', 'experience_level']]
    y = df['job_role']
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('skills_emb', SkillsEmbeddingTransformer(embedding_dim=16), ['skills']),
            ('qual_ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['qualification']),
            ('exp_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['experience_level']),
        ],
        remainder='drop' 
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', CandidateMatcherEnsemble(use_stacking=True))
    ])
    
    print("Debug: Fitting model (this might take time)...")
    pipeline.fit(X_train, y_train)
    print("Debug: Model fitted.")
    
    print("Debug: Saving artifacts...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/production_pipeline.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    print("Debug: Done.")

if __name__ == "__main__":
    main()
