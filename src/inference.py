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
    # Checking for the artifact. In K8s, this passes via the 'MODEL_PATH' env var.
    default_path = "models/production_pipeline.pkl"
    model_path = os.getenv("MODEL_PATH", default_path)
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
        
        # Determine top match initially
        top_match = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[top_match]
        
        # [Misem]: Algorithmic Fallback Pattern.
        # If the model is unsure (confidence < 0.4), we don't trust it. 
        # We fall back to a simple keyword heuristic (Rule-based).
        # This prevents the system from making wild guesses on unfamiliar data.
        if confidence < 0.4:
            print(f"Confidence {confidence} below threshold. Triggering Fallback.")
            # Simple Heuristic: Check if 'SQL' or 'Python' is in skills -> Likely Data role
            s = candidate.skills.lower()
            if 'sql' in s or 'python' in s or 'data' in s:
                fallback_role = "Data Scientist" # Simplified guess
            else:
                fallback_role = "Unknown"
            
            top_match = fallback_role
            confidence = 0.5 # Heuristic confidence
            prob_dict = {fallback_role: 0.5, "Model_Low_Conf": 0.5}

        # [Anas]: Batch Serving Note.
        # We process single requests here (Stateless Serving).
        # If we had to score 1 Million candidates nightly, we'd switch to BATCH SERVING (e.g. Apache Spark/Airflow).
        # Batch is better for throughput, but this API is built for Latency (Real-time).
        
        # [Mohammed Ali]: CME Data Logging.
        # We save the features + prediction to a log. 
        # The 'monitoring.py' service (or a scheduled job) picks this up to run Great Expectations drift checks.
        with open("inference_log.csv", "a") as f:
            f.write(f"{candidate.candidate_id},{top_match},{confidence}\n")

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
