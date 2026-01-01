from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import mlflow.sklearn
import os

app = FastAPI(title="Candidate Matching Service", version="1.0.0")

# Input Schema
class CandidateInput(BaseModel):
    candidate_id: str
    skills: str
    qualification: str
    experience_level: str

# Output Schema
class PredictionOutput(BaseModel):
    candidate_id: str
    job_role_probabilities: dict
    top_match: str
    confidence: float

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    global model
    # [Anas]: Bootstrapping the model. 
    # Checking for the artifact locally. In K8s, we'd mount this volume or pull from S3.
    model_path = "models/production_pipeline.pkl"
    if os.path.exists(model_path):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("Warning: No model found at startup. Service will fail predictions.")

@app.post("/predict", response_model=PredictionOutput)
def predict(candidate: CandidateInput):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Create DataFrame from input
    data = pd.DataFrame([candidate.dict()])
    
    # Predict Probabilities
    try:
        # [Misem]: This is the key "Reframing" pattern.
        # We output the full probabilities so we can decide later if it's a "Maybe".
        probs = model.predict_proba(data)[0] 
        classes = model.classes_
        
        prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
        
        # Determine top match
        top_match = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[top_match]
        
        return PredictionOutput(
            candidate_id=candidate.candidate_id,
            job_role_probabilities=prob_dict,
            top_match=top_match,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def feedback(candidate_id: str, actual_role: str):
    # [Misem]: Responsible AI pattern.
    # Collecting the ground truth so we can check if we were actually right later.
    with open("feedback_log.csv", "a") as f:
        f.write(f"{candidate_id},{actual_role}\n")
    return {"status": "received"}
