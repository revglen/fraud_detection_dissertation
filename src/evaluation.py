import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report 
from utils import Log_Experiment
from config import EXPERIMENT_LOG

class Evaluation:

    @staticmethod
    def read_experiment_log(path=None):
        p = path or EXPERIMENT_LOG
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run experiments first.")
        df=pd.read_csv(p)
        return df

    @staticmethod
    def compute_metrics_for_model(df, model_name):
        subset = df[df['model_name']==model_name].copy()
        if subset.empty:
            return None
        
        y_true=subset['model_name'].apply(Log_Experiment.label_to_int)
        y_pred_raw = subset['predicted_label']
        y_pred = y_pred_raw.apply(Log_Experiment.label_to_int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        return {
            'model_name': model_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'support': int(y_true.sum())
        }

    @staticmethod
    def compare_models(model_names, log_path=None):
        """
        Compute metrics for a list of model_names and return a simple DataFrame.
        """
        df = Evaluation.read_experiment_log(log_path)
        rows = []
        for name in model_names:
            m = Evaluation.compute_metrics_for_model(df, name)
            if m:
                rows.append(m)
        return pd.DataFrame(rows)

