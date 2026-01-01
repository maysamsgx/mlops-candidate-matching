import mlflow
import mlflow.sklearn
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import joblib
import os

class CandidateMatcherEnsemble(BaseEstimator, ClassifierMixin):
    """
    [Misem]: This is the heavy hitter. 
    I'm stacking XGBoost, LGBM, and Random Forest because single models weren't cutting it for the complex cases.
    It's a trade-off: training is slower, but the resilience to outliers is way better.
    """
    def __init__(self, use_stacking=True):
        self.use_stacking = use_stacking
        self.clf = None
        
    def _get_base_models(self):
        # [Ele & Eman]: Tweaked these params after about 50 runs.
        # XGBoost needs the depth constraint (8) to stop overfitting.
        xgb_clf = xgb.XGBClassifier(
            max_depth=8, 
            learning_rate=0.05, 
            n_estimators=300, 
            subsample=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42
        )
        
        # [Ele]: LGBM is our speed demon here. 64 leaves felt like the sweet spot.
        lgb_clf = lgb.LGBMClassifier(
             num_leaves=64, 
            learning_rate=0.05, 
            n_estimators=300, 
            random_state=42
        )
        
        # [Misem]: Keeping RF as the stable fallback. It's less sensitive to the hyperparams than the boosters.
        rf_clf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=None,
            max_features='sqrt',
            random_state=42
        )
        
        return [
            ('xgb', xgb_clf),
            ('lgb', lgb_clf),
            ('rf', rf_clf)
        ]
        
    def fit(self, X, y):
        estimators = self._get_base_models()
        
        if self.use_stacking:
            # [Misem]: Stacking > Voting. 
            # Letting LogisticRegression figure out who to trust (Meta-learner).
            self.clf = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                stack_method='predict_proba',
                cv=3
            )
        else:
            # Fallback to simple Soft Voting if stacking proves too unstable or slow.
            self.clf = VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
            
        self.clf.fit(X, y)
        return self
        
    def predict(self, X):
        return self.clf.predict(X)
        
    def predict_proba(self, X):
        # [Misem]: We need the full probability spread, not just the label.
        # Essential for the "Risk-Aware" requirement.
        return self.clf.predict_proba(X)

def save_checkpoint(model, path, epoch=None):
    """
    [Misem]: Safety net. Using this to dump state in case the training pod dies.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    if epoch:
        print(f"Checkpoint saved for epoch {epoch} at {path}")

def log_metrics_to_mlflow(y_true, y_pred, y_proba):
    """
    Logs comprehensive metrics to MLflow.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
    
    acc = accuracy_score(y_true, y_pred)
    # Handle multiclass AUC
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
    except:
        auc = 0.0 # Fallback
        
    loss = log_loss(y_true, y_proba)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc_roc", auc)
    mlflow.log_metric("log_loss", loss)
    
    return {"accuracy": acc, "auc": auc, "loss": loss}
