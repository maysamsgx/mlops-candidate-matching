import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import hashlib

class SkillsEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    [Misem]: Rolled our own embedding logic here. One-hot encoding was exploding the memory with these skill sets, 
    so I mapped them to dense vectors (16-dim). It's a bit custom, but handles the cardinality way better.
    """
    def __init__(self, embedding_dim=16, max_vocab_size=1000):
        self.embedding_dim = embedding_dim
        self.max_vocab_size = max_vocab_size
        self.embedding_matrix = None
        self.vocab = {}
    
    def fit(self, X, y=None):
        # Establish the vocabulary from the training data.
        # We need to handle potential new skills in production, but for now, we'll just ignore unknown ones.
        if hasattr(X, 'iloc'):
             X = X.iloc[:, 0]
        unique_skills = set()
        for text in X:
            if pd.isna(text): continue
            skills = [s.strip() for s in text.split(',')]
            unique_skills.update(skills)
        
        # Sort for reproducibility
        sorted_skills = sorted(list(unique_skills))
        if len(sorted_skills) > self.max_vocab_size:
            sorted_skills = sorted_skills[:self.max_vocab_size]
            
        self.vocab = {skill: i+1 for i, skill in enumerate(sorted_skills)} # 0 is padding/unknown
        
        # [Misem]: Initializing random vectors. 
        # Ideally we'd pull GloVe or Word2Vec here, but since we're training from scratch for the course constraints,
        # random projection works as a solid baseline. Keeps it lightweight for the team to run locally.
        np.random.seed(42)
        self.embedding_matrix = np.random.normal(size=(len(self.vocab) + 1, self.embedding_dim))
        return self

    def transform(self, X):
        if hasattr(X, 'iloc'):
             X = X.iloc[:, 0]
        
        embeddings = []
        for text in X:
            if pd.isna(text):
                embeddings.append(np.zeros(self.embedding_dim))
                continue
            
            skills = [s.strip() for s in text.split(',')]
            skill_indices = [self.vocab.get(s, 0) for s in skills]
            
            if not skill_indices:
                embeddings.append(np.zeros(self.embedding_dim))
            else:
                # Average embedding
                skill_vecs = self.embedding_matrix[skill_indices]
                embeddings.append(np.mean(skill_vecs, axis=0))
                
        return np.vstack(embeddings)

class HashedSkillsTransformer(BaseEstimator, TransformerMixin):
    """
    [Ahmed AlHayek]: Fallback plan.
    If Misem's embeddings get too heavy, we swap to this. 
    It's standard hashing - fast, no memory overhead. Good for the baseline comparison.
    """
    def __init__(self, n_features=20):
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # HashingVectorizer expects iterable of strings
        # We treat standard comma-sep strings as documents
        return self.vectorizer.transform(X).toarray()

class FeatureCrossTransformer(BaseEstimator, TransformerMixin):
    """
    [Misem]: Explicitly crossing 'experience_level' with 'skills_count'.
    I noticed in EDA that seniors with few skills vs juniors with many skills behave differently.
    Capturing this interaction manually boosts the tree models.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Expects dataframe with 'experience_level' and 'skills'
        X_out = X.copy()
        
        # Derived feature: Skills Count
        X_out['skills_count'] = X_out['skills'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
        
        # [Misem]: Binning the counts mostly to reduce noise. 
        # 'Expert' with 100+ skills is probably a data error, but we'll cap it at 'expert' bucket.
        X_out['exp_skills_cross'] = X_out['experience_level'].astype(str) + "_" + pd.cut(X_out['skills_count'], bins=[0, 3, 6, 10, 100], labels=['low', 'med', 'high', 'expert']).astype(str)
        
        return X_out[['exp_skills_cross']]

class PreprocessingPipeline:
    def __init__(self):
        pass
        
    # We will build standard sklearn pipeline in the workflow script
