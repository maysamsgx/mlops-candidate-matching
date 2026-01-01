# Candidate Matching MLOps System

**GitLab Repository**: [https://gitlab.com/maysamsgx/candidate-matching-system](https://gitlab.com/maysamsgx)

This project implements a **Level 2 MLOps** candidate matching system with a focus on high-cardinality feature handling, ensemble modeling, and continuous evaluation.

## 👥 Team Role Mapping & Contributions

| Role | Team Member | Responsible For |
|------|-------------|-----------------|
| **Project Manager & Lead Data Scientist** | **Misem Mohamed** | **ALL New Implementations & Enhancements**: <br> • New Embeddings / Feature Crosses <br> • Advanced Ensemble Logic (Stacking) <br> • CME Enhancements & Business Logic <br> • New ML Design Patterns <br> • GitLab CI/CD Integration <br> • MLflow enhancements |
| **DevOps Engineer & Business Analyst** | Anas Brkji | CI/CD Pipeline Base, Docker/Kubernetes Base |
| **Test Engineer & Business Analyst** | Ahmed A.S Abubreik | Unit Testing Framework, Requirements |
| **Data Engineer** | Ahmed N.F AlHayek | Data Ingestion Base |
| **MLOps SRE** | Mohammed Ali | Monitoring Infrastructure Base |
| **ML Engineer (Model Development)** | Ele Ben Messaoud | Base Model Development |
| **ML Engineer (Optimization & Tuning)** | Eman Mohammed | Hyperparameter Tuning Base |

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv .venv
# Activate: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux)
pip install -r requirements.txt
```

### 2. Run the Training Pipeline (Prefect)
This command will ingest data, validate it (Great Expectations), train the ensemble model (XGBoost/LGBM/RF), and save artifacts.
```bash
python src/workflow.py
```
*Artifacts will be saved to `models/`.*

### 3. Serve the Model (FastAPI)
Start the stateless inference server:
```bash
uvicorn src.inference:app --reload
```
Test with curl (PowerShell):
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{"candidate_id": "C001", "skills": "Python, SQL, Terraform", "qualification": "MSc", "experience_level": "Senior"}'
```

### 4. Monitor Experiments (MLflow)
View experiment runs and metrics:
```bash
mlflow ui
```

## 🏗 System Architecture

### Pillars & Patterns
- **Data**: Embeddings for `skills` (16-dim), Feature Crosses (`exp_level` x `skills_count`).
- **Modeling**: Stacking Ensemble (XGB+LGBM+RF -> LogReg). Outputs probability distributions (Reframing).
- **Resilience**: Stateless FastAPI service, Circuit Breakers (simulated via fallbacks).
- **One-Click Run**: Created **`run_system_e2e_demo.bat`**.
  - Your team just needs to double-click this (or run it in terminal).
  - It automatically checks Python, creates the `.venv`, installs dependencies, trains the model, and starts the server. No permissions/setup headaches.
- **Verification**: Verified relative paths so cloning into any folder works.
- **CME**: Drift detection using Great Expectations logic in `src/monitoring.py`.

### CI/CD
- **Pipeline**: `.gitlab-ci.yml` defines the Test -> Build -> Deploy stages.
- **Docker**: `Dockerfile` provided for containerization.

### 🏁 Final Handover
The project is clean (`my_venv` gone), fully tested, and documented.
You can simply commit this to your GitLab, and your team can check it out and run `run_system_e2e_demo.bat` immediately.

## 📂 Project Structure
- `src/`: Source code for pipelines and services.
- `tests/`: Unit tests.
- `models/`: Saved model artifacts.
- `.gitlab-ci.yml`: CI/CD configuration.
