import os
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from utils import Log_Experiment
from joblib import dump, load
from config import RANDOM_STATE

class ML_Pipeline:
    @staticmethod
    def get_feature_columns(df):
        return ['amount', 'log_amount', 'hour', 'is_high_amount']
    
    def train_ml_model(X_train, y_train, model_type="rf", random_state=RANDOM_STATE):
        feature_cols=ML_Pipeline.get_feature_columns(X_train)
        numeric_transformer=StandardScaler()
        preprocessor  = ColumnTransformer(transformers=[('num', numeric_transformer, feature_cols)])

        if model_type=="rf":
            clf=RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
        elif model_type=="logreg":
            clf=LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced')
        else:
            raise ValueError("model_type must be 'rf' or 'logreg'")

        pipeline=Pipeline(steps=[('preprocessor', preprocessor), ('clf', clf)])
        pipeline.fit(X_train[feature_cols], y_train)
        return pipeline
        
    @staticmethod
    def predict_and_log(pipeline, X, y_true, model_name="RandomForest", prompt_variant=None):
        feature_cols=ML_Pipeline.get_feature_columns(X)
        start=time.time()

        probs = pipeline.predict_proba(X[feature_cols])[:,1]        
        preds=(probs >= 0.5).astype(int)

        latency=(time.time()-start)/len(X)
        precision=precision_score(y_true, preds)
        recall=recall_score(y_true, preds)
        f1=f1_score(y_true, preds)
        cm=confusion_matrix(y_true, preds)

        rows=[]
        for idx, tx_id in enumerate(X['transaction_id'].tolist()):
            rows.append({
                    'transaction_id': tx_id,
                    'true_label': int(y_true.iloc[idx]),
                    'predicted_label': int(preds[idx]),
                    'predicted_prob': float(probs[idx]),
                    'model_name': model_name,
                    'prompt_variant': prompt_variant or '',
                    'reason': '',  # ML reasons will be added separately (e.g., SHAP)
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })

        Log_Experiment.append_experiment_log(rows)

        metrics={
        'model_name': model_name,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'latency_per_tx': latency
        }

        return metrics

    @staticmethod
    def get_feature_importance(pipeline, feature_names=None, top_k=10):
        clf=pipeline.named_steps['clf']
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            names = feature_names
        elif hasattr(clf,"coef_"):
            importances=np.abs(clf.cf_).ravel()
            names=feature_names
        else:
            return pd.DataFrame()

        df=pd.DataFrame({'feature': names, 'importance': importances})
        return df.sort_values('importance', ascending=False).head(top_k)