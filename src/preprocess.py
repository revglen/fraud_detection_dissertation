import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import CREDITCARD_CSV, RANDOM_STATE
from utils import Log_Experiment

class Preporcessing:

  @staticmethod
  def load_creditcard(path=CREDITCARD_CSV):

    #Check for valid path
    if len(path.strip()) <= 0:
      raise Exception ("The Path is empty")

    absolute_path = path
    if not os.path.isabs(path):
      absolute_path = os.getcwd() + absolute_path

    print(absolute_path)
    file = Path(absolute_path)    
    
    if not file.exists():
      raise Exception ("Path does not exists")
      
    df=pd.read_csv(absolute_path)
    if 'Class' in df.columns:
      df=df.rename(columns={'Class': 'label'})

    if 'transaction_id' not in df.columns:
      df=df.reset_index().rename(columns={'index': 'transaction_id'})

    df['txn_text'] = df.apply(lambda r: f"Transaction TID: {r['transaction_id']} — Time: {r.get('Time','NA')} — Amount: ${r.get('Amount','NA')}", axis=1)
    df['label'] = df['label'].apply(Log_Experiment.label_to_int)
    return df

  @staticmethod
  def basic_feature_engineering(df):
    df=df.copy()
    if 'Amount' in df.columns or 'TransactionAmt' in df.columns:
      amt_col='Amount' if 'Amount' in df.columns else 'TransactionAmt'
      df['amount'] =pd.to_numeric(df[amt_col], errors='coerce').fillna(0.0)
    else:
      df['amount']=0.0

    df['log_amount'] = np.log1p(df['amount'])

    if 'Time' in df.columns:
      try:
        df['hour'] = (df['Time']//3600) % 24
      except:
        df['hour'] = 0
    else:
        df['hour'] = 0

    df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)

    ml_cols = ['transaction_id', 'amount', 'log_amount', 'hour', 'is_high_amount', 'label', 'txn_text']
    for c in ml_cols:
      if c not in df.columns:
        df[c]=0

    return df[ml_cols]
  
  @staticmethod
  def train_val_split(df, test_size=0.15, val_size=0.15, random_state=RANDOM_STATE):
    df=df.copy()
    strat_col='label' if 'label' in df.columns else df.iloc[:,-1].name
    
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[strat_col], random_state=random_state)
    val_fraction=val_size / (1-test_size)
    train, val = train_test_split(train_val, test_size=val_fraction, stratify=train_val[strat_col], random_state=random_state)
    
    X_train=train.drop(columns=['label']).reset_index(drop=True)
    X_val=val.drop(columns=['label']).reset_index(drop=True)
    X_test=test.drop(columns=['label']).reset_index(drop=True)

    y_train=train['label'].reset_index(drop=True)
    y_val=val['label'].reset_index(drop=True)
    y_test=test['label'].reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test
