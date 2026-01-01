import pytest
import numpy as np
from src.modeling import CandidateMatcherEnsemble

def test_ensemble_initialization():
    model = CandidateMatcherEnsemble(use_stacking=True)
    assert model.use_stacking is True
    assert model.clf is None

def test_ensemble_fit_predict():
    # Mock data
    X = np.random.rand(20, 10)
    y = np.random.randint(0, 3, size=20)
    
    model = CandidateMatcherEnsemble(use_stacking=False) # Use voting for faster test
    model.fit(X, y)
    
    preds = model.predict(X)
    assert len(preds) == 20
    
    probs = model.predict_proba(X)
    assert probs.shape == (20, 3)
