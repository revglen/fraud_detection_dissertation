import os
import joblib
import pandas as pd
from preprocess import Preporcessing
from ml_pipeline import ML_Pipeline
from llm_pipeline import LLM_Pipeline
from evaluation import Evaluation
from config import RANDOM_STATE, LOCAL_MISTRAL_ID, LOCAL_LLAMA_ID

class Experiment:
    """
        Orchestration script to run the full experiment (ML + LLM) in order:
        1. Load & preprocess
        2. Train ML models
        3. Evaluate ML on test set and log
        4. Optionally build RAG index
        5. Run LLM (with or without RAG) and log
        6. Produce a small summary of results

        This script is intended to be run from the repository root:
        python -m src.experiment
    """

    @staticmethod
    def run_full_experiment(dataset="creditcard", use_rag=False, use_local_llm=True):
        print("Loading Dataset...")
        if dataset == "creditcard":
            df_raw=Preporcessing.load_creditcard()

        print("Feature engineering...")
        df=Preporcessing.basic_feature_engineering(df_raw)
        print("Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test=Preporcessing.train_val_split(df)

        # ML: Random Forest
        print("Training Random Forest...")
        rf_pipeline=ML_Pipeline.train_ml_model(X_train, y_train, model_type="rf", random_state=RANDOM_STATE)
        rf_metrics = ML_Pipeline.predict_and_log(rf_pipeline, X_test, y_test, model_name="RandomForest")
        if not os.path.exists("models"):
            os.makedirs("models")
        joblib.dump(rf_pipeline, "models/rf_pipeline.joblib")

        # ML: Logistic Regression
        print("Training Logistic Regression...")
        log_pipeline = ML_Pipeline.train_ml_model(X_train, y_train, model_type='logreg', random_state=RANDOM_STATE)
        log_metrics = ML_Pipeline.predict_and_log(log_pipeline, X_test, y_test, model_name="LogisticRegression")
        joblib.dump(log_pipeline, "models/logreg_pipeline.joblib")

        rag_index = None
        if use_rag:
            print("Building vector index for RAG (this requires langchain & chromadb)...")
            combined_df = pd.concat([X_train, X_val], ignore_index=True)
            docs = [(row['transaction_id'], row['txn_text']) for _, row in combined_df.iterrows()]
            rag_index=LLM_Pipeline.build_rag_index(docs, persist_dir="vector_store")

        # LLM runs on test dataset
        print("Preparing test DataFrame for LLM...")
        llm_test_df=X_test[['transaction_id', 'txn_text']].copy()
        llm_test_df['label'] = y_test.values
        
        if use_local_llm:
            print("Running Mistralai on test set...")
            LLM_Pipeline.huggingface_login()

            # Fine-tune first - Transfer Knowledge 
            tuned_model_path = LLM_Pipeline.fine_tune_llm_on_fraud_data(
                model_name=LOCAL_LLAMA_ID,
                train_df=llm_test_df,          # same split you use for ML
                num_train_epochs=1
            )

            llm_results = LLM_Pipeline.run_local_llm_on_df(llm_test_df, engine="local_llama",  model_identifier=tuned_model_path,
                                prompt_type="base", use_rag=use_rag, rag_collection=rag_index,
                                max_new_tokens=64, temperature=0.0, device="cpu")
        else:
            print("Running OpenAI ChatGPT on test set...")
            llm_results = LLM_Pipeline.run_llm_on_df(llm_test_df, engine="openai", model="gpt-4o-mini", prompt_type="base", use_rag=use_rag, rag_index=rag_index, temperature=0.0)
            
        # Summarise
        print("Comparing model metrics...")
        models_to_compare = ["RandomForest", "LogisticRegression", "LLM_openai_gpt-4o-mini"]
        # note: LLM model_name in logs uses pattern "LLM_openai_gpt-4o-mini"
        
        comp = Evaluation.compare_models(models_to_compare)
        print(comp)

if __name__ == "__main__":
    # By default run creditcard experiment without RAG
    Experiment.run_full_experiment(dataset='creditcard', use_rag=True, use_local_llm=True)

