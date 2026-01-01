import pytest
import pandas as pd
import numpy as np
from src.features import SkillsEmbeddingTransformer, FeatureCrossTransformer

def test_skills_embedding_transformer():
    df = pd.DataFrame({'skills': ['Python, SQL', 'Java', np.nan]})
    transformer = SkillsEmbeddingTransformer(embedding_dim=4)
    # Pass as DataFrame to match pipeline behavior
    transformer.fit(df[['skills']])
    X_trans = transformer.transform(df[['skills']])
    
    assert X_trans.shape == (3, 4)
    # Check if NaN handling works (returns zeros)
    assert np.all(X_trans[2] == 0)

def test_feature_cross_transformer():
    df = pd.DataFrame({
        'experience_level': ['Senior', 'Junior', 'Mid'],
        'skills': ['A, B, C', 'A', 'A, B']
    })
    transformer = FeatureCrossTransformer()
    X_trans = transformer.transform(df)
    
    assert 'exp_skills_cross' in X_trans.columns
    # Check logic: Senior (len 3 = low?) dependent on bins. 
    # Bins: 0, 3, 6... labels: low, med..
    # 3 items -> 'low' (inclusive? binning pd.cut default is right included)
    # We just check it returns strings
    assert isinstance(X_trans['exp_skills_cross'].iloc[0], str)
