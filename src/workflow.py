from prefect import task, flow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import mlflow
import joblib
import os
from src.features import SkillsEmbeddingTransformer, FeatureCrossTransformer, HashedSkillsTransformer
from src.modeling import CandidateMatcherEnsemble, log_metrics_to_mlflow
from src.monitoring import DataValidator

@task
def load_data(path: str):
    df = pd.read_csv(path)
    # Filter rare classes to avoid CV errors in Stacking
    v = df['job_role'].value_counts()
    df = df[df['job_role'].isin(v[v >= 3].index)]
    return df

@task
def validate_data(df: pd.DataFrame):
    validator = DataValidator()
    # [Mohammed Ali]: Just logging warnings for the demo.
    # In Prod, we'd raise an exception here and stop the pipeline to prevent pollution.
    result = validator.validate(df)
    if not result['success']:
        print(f"Validation WARNING: {result}")
    return df

@task
def train_model(df: pd.DataFrame):
    # Split
    X = df[['candidate_id', 'skills', 'qualification', 'experience_level']]
    y = df['job_role']
    
    # Encoder for target
    # We need to map job_role strings to integers for XGB/LGBM usually, 
    # but the wrappers might handle it. Safest is to encode y.
    # However, for simplicity here we rely on the estimators handling it or add a label encoder.
    # We'll assume the ensemble handles string labels (RandomForest does not in sklearn, need LabelEncoder).
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    
    # [Misem]: Stacking logic simplified regarding columns.
    # We kept the FeatureCross separate in 'features.py' logic, here we stick to the main transformers.
    
    # Skills -> Embedding (My custom one)
    # Qual -> Ordinal
    # Exp -> OneHot
    # Cross -> Hashing (since 'exp_skills_cross' is high cardinality string)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('skills_emb', SkillsEmbeddingTransformer(embedding_dim=16), ['skills']),
            ('qual_ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['qualification']),
            ('exp_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['experience_level']),
            # We add the cross generation and hashing as a separate pipeline branch?
            # It's tricky with basic ColumnTransformer.
            # We will perform Feature Cross separately or assume it's handled.
            # Let's use the FeatureCrossTransformer in a separate pipeline that generates the column,
            # then hashes it.
            # BUT ColumnTransformer takes columns.
            # We'll skip complex pipeline branching for this demo and focus on the main embeddings.
        ],
        remainder='drop' 
    )
    
    # Full Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', CandidateMatcherEnsemble(use_stacking=True))
    ])
    
    # MLflow Tracking
    mlflow.set_experiment("candidate_matching_experiment")
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        
        metrics = log_metrics_to_mlflow(y_test, y_pred, y_proba)
        
        # Log Params
        mlflow.log_params({
            "model_type": "StackingEnsemble",
            "feature_engineering": "Embeddings+OHE"
        })
        
        # Save Model
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Register Model (Simulated)
        # mlflow.register_model(f"runs:/{run_id}/model", "CandidateMatcher")

    return pipeline, le

@task
def save_for_serving(pipeline, label_encoder):
    # [Anas]: Dumping the binary. 
    # Ops team picks this up for the container build.
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/production_pipeline.pkl")
    # Need the label encoder to decode the 0s and 1s back to 'Data Scientist' etc.
    joblib.dump(label_encoder, "models/label_encoder.pkl")

@flow(name="candidate-matching-pipeline")
def main_flow():
    data_path = "candidate_job_role_dataset.csv"
    if not os.path.exists(data_path):
        print("Dataset not found!")
        return
        
    df = load_data(data_path)
    df_validated = validate_data(df)
    pipeline, le = train_model(df_validated)
    save_for_serving(pipeline, le)

if __name__ == "__main__":
    main_flow()
