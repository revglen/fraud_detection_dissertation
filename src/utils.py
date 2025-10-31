import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from config import EXPERIMENT_LOG, LOG_DIR

class Log_Experiment:

    @staticmethod
    def ensure_log_dir():
         #Check for valid path
        if len(EXPERIMENT_LOG.strip()) <= 0:
            raise Exception ("The Path is empty")

        absolute_path = EXPERIMENT_LOG
        if os.path.isabs(absolute_path):
            absolute_path = os.getcwd() + absolute_path

        print(absolute_path)

        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

    @staticmethod
    def append_experiment_log(df_rows):
        Log_Experiment.ensure_log_dir()
        columns =['transaction_id', 'true_label', 'predicted_label', 'predicted_prob',
                    'model_name', 'prompt_variant', 'reason', 'timestamp']
        df=pd.DataFrame(df_rows, columns=columns)
        if os.path.exists(EXPERIMENT_LOG):
            df.to_csv(EXPERIMENT_LOG, mode="a", header=False, index=False)
        else:
            df.to_csv(EXPERIMENT_LOG, index=False)
            
    @staticmethod
    def label_to_int(label):
        if isinstance(label, (int, float)):
            return int(label)

        s=str(label).strip().lower()
        if s in ['1', 'true', 'fraud', 'fraudulent', 'yes']:
            return 1
        
        return 0
    
    @staticmethod
    def safe_parse_llm_response(text):
        label, conf, reason=None, None, None
        if not text:
            return None, None, None

        lines = [l.strip for l in text.splitlines() if l.strip()]
        for line in lines:
            if line.lower().startswidth("label"):
                try:
                    label = line.split(":",1)[1].strip()
                except:
                    label = line
            elif line.lower().startswidth("confidnce"):
                try:
                    conf = float(line.split(":",1)[1].strip())
                except:
                    conf = None
            elif line.lower().startswidth("reason"):
                try:
                    reason = line.split(":",1)[1].strip()
                except:
                    reason = line

        # fallback attempts
        if label is None and len(lines)>=1:
            label = lines[0]
        if reason is None and len(lines)>=1:
            reason = lines[-1]
        return label, conf, reason

#   @staticmethod
#   def log_predictions(transactionsId, y_true, y_pred, probs=None, model_name="ML_model", prompt_variant=None):
#         if not os.path.exists(LOG_FILE):
#             os.makedirs(LOG_FOLDER)

#         df=pd.DataFrame({
#             'transaction_id': transactionsId,
#             'true_label': y_true,
#             'predicted_label': y_pred,
#             'prediction_prob': probs if probs is not None else [None]*len(y_pred),
#             'model_name': model_name,
#             'prompt_variant': prompt_variant
#         })

#         if os.path.exists(LOG_FILE):
#             df.to_csv(LOG_FILE, mode='a', header=False, index=False)
#         else:
#             df.to_csv(LOG_FILE, index=False)

       